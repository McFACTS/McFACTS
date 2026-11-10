#!/usr/bin/env python3
"""Test the SettingsHandler object"""
######## Imports ########
#### Standard Library ####
import concurrent.futures
from copy import deepcopy, copy
import pickle
import time
import warnings

#### Third Party ####
import tqdm
import numpy as np
import pytest

#### Local ####
from mcfacts.inputs import settings_manager as sm
from mcfacts.inputs.settings_manager import (
    SettingsManager, SettingsProperty, StaticSettingsProperty, OptionalSettingsProperty,
    CategoryProxy, DEFAULT_SETTINGS,
)

######## Setup ########
N_PROC = 10

def monkey_hit_head(settings):
    assert isinstance(settings, SettingsManager)
    rng = np.random.Generator(np.random.Philox(seed=settings.seed))
    time.sleep(rng.random())
    return rng.choice([True, False])


# region SettingsProperty Initialization

# Make sure all of the fields are stored correctly and that the aliases are stored as a list
def test_property_stores_fields():
    prop = SettingsProperty("name", "cat", 1, int, aliases=["y"])
    assert (prop.name, prop.category, prop.value, prop.prop_type, list(prop.aliases)) == ("name", "cat", 1, int, ["y"])


# Test that the default of the alias list is an empty
def test_property_aliases_default_empty():
    assert list(SettingsProperty("name", "cat", 1, int).aliases) == []


# Test that we reject blank or pure whitespace property names
@pytest.mark.parametrize("name", ["", "  "])
def test_property_rejects_blank_name(name):
    with pytest.raises(ValueError):
        SettingsProperty(name, "cat", 1, int)


# Test that we reject blank or pure whitespace category names
@pytest.mark.parametrize("category", ["", "  "])
def test_property_rejects_blank_category(category):
    with pytest.raises(ValueError):
        SettingsProperty("name", category, 1, int)


# Check that we reject a value that is not the same as the input type
def test_property_rejects_wrong_type():
    with pytest.raises(TypeError):
        # Value is float, but int required
        SettingsProperty("name", "cat", 1.5, int)


# Check That we specifically reject bools as ints -- bools should be bools!
def test_property_rejects_bool_as_int():
    with pytest.raises(TypeError):
        SettingsProperty("name", "cat", True, int)


# Check that we reject a single string as an alias since its supposed to be a list of strings
def test_property_rejects_str_aliases():
    # Reject if alias is just a string and not a list
    with pytest.raises(TypeError):
        SettingsProperty("name", "cat", 1, int, aliases="ab")


# Check that we reject empty or pure whitespace strings in the alias list
@pytest.mark.parametrize("alias", ["", "  "])
def test_property_rejects_blank_aliases(alias):
    with pytest.raises(ValueError):
        SettingsProperty("x", "c", 1, int, aliases=[alias])


# Check that mutating the caller's alias list after construction doesn't change the property
@pytest.mark.parametrize("signature", [SettingsProperty, StaticSettingsProperty])
def test_property_aliases_not_shared_with_caller(signature):
    aliases = ["y"]
    prop = signature("name", "cat", 1, int, aliases=aliases)
    aliases.append("z")
    assert tuple(prop.aliases) == ("y",)


# Test that cached properties are stored correctly and that the aliases are stored as a list
def test_static_property_stores_fields():
    prop = StaticSettingsProperty("name", "cat", 1, int, aliases=["y"])
    assert (prop.name, prop.category, prop.value, prop.prop_type, list(prop.aliases)) == ("name", "cat", 1, int, ["y"])


# Check that a static property's value cannot be reassigned
def test_static_property_value_is_read_only():
    prop = StaticSettingsProperty("name", "cat", 1, int)
    with pytest.raises(AttributeError):
        prop.value = 5


# Check that properties with different aliases are not equal
def test_property_not_equal_different_aliases():
    assert SettingsProperty("name", "cat", 1, int, aliases=["y"]) != SettingsProperty("name", "cat", 1, int)

# endregion


# region SettingsProperty Equal & Serialization

# Check that the string representation of a property is "category.name"
def test_property_to_str():
    assert str(SettingsProperty("name", "cat", 1, int)) == "cat.name"


# Check that two properties with the same fields are equal
def test_property_equal():
    prop1 = SettingsProperty("name", "cat", 1, int, aliases=["y"])
    prop2 = SettingsProperty("name", "cat", 1, int, aliases=["y"])

    assert prop1 == prop2


# Check that two properties with different, but compatible, types are not equal
def test_property_not_equal_to_other_type():
    prop1 = SettingsProperty("name", "cat", 1, int, aliases=["y"])
    prop2 = SettingsProperty("name", "cat", True, bool, aliases=["y"])

    assert prop1 != prop2


# Check that we can property pickle and unpickle a SettingsProperty
def test_property_pickle():
    prop = SettingsProperty("name", "cat", 1, int, aliases=["y"])
    pickled = pickle.dumps(prop)
    unpickled = pickle.loads(pickled)
    assert prop == unpickled


# Check that we can copy a SettingsProperty and that the copy is equal but not the same object
def test_property_copy():
    prop = SettingsProperty("name", "cat", 1, int, aliases=["y"])
    prop_copy = copy(prop)
    assert prop == prop_copy and prop is not prop_copy

# endregion


# region Built-in DEFAULT_SETTINGS Validation

# Check that all of the defined property keys are unique
def test_default_keys_are_unique():
    keys = [key for p in DEFAULT_SETTINGS for key in (p.name, *p.aliases)]
    assert len(keys) == len(set(keys))


# Check that categories and keys don't overlap
def test_default_categories_dont_overlap_with_keys():
    assert not {p.category for p in DEFAULT_SETTINGS} & sm.DEFAULT_KEYS


# Check that aliases and keys don't overlap
def test_default_aliases_dont_overlap_with_keys():
    aliases = [alias for p in DEFAULT_SETTINGS for alias in p.aliases]
    assert not set(aliases) & sm.DEFAULT_KEYS


# Make sure the property keys don't shadow direct attributes of the SettingsManager class
def test_default_keys_not_shadowed_by_manager_attributes():
    assert not set(dir(SettingsManager)) & sm.DEFAULT_KEYS


# Test to make sure default property aliases don't conflict with the base keys
def test_default_aliases_dont_conflict_with_manager_attributes():
    aliases = [alias for p in DEFAULT_SETTINGS for alias in p.aliases]
    assert not set(dir(SettingsManager)) & set(aliases)

# endregion


# region CategoryProxy Initialization

# Test that the default values of a CategoryProxy are empty dicts for props and aliases
def test_proxy_defaults_empty():
    proxy = CategoryProxy()
    assert proxy.props == {} and proxy.aliases == {}


# Test that we reject aliases with the same name as a property
def test_proxy_rejects_alias_matching_prop():
    with pytest.raises(AttributeError):
        CategoryProxy({"a": 1, "b": 2}, {"a": "b"})


# Test that we reject aliases for properties that don't exist in the props dict
def test_proxy_rejects_alias_to_missing_prop():
    with pytest.raises(AttributeError):
        CategoryProxy({"a": 1}, {"b": "missing"})


# Test that we reject property names that are the same as the class attributes
@pytest.mark.parametrize("name", ["props", "aliases"])
def test_proxy_reject_exist_attribute_by_name(name):
    with pytest.raises(AttributeError):
        CategoryProxy({name: 1})

# Test that we reject aliases that are the same as the class attributes
@pytest.mark.parametrize("alias", ["props", "aliases"])
def test_proxy_reject_existing_attribute_by_alias(alias):
    with pytest.raises(AttributeError):
        CategoryProxy({"a": 1}, {alias: "a"})


# Check that changing the caller's props dict after construction doesn't change the proxy
def test_proxy_props_not_shared_with_caller():
    props = {"a": 1}
    proxy = CategoryProxy(props)
    props["a"] = 2
    assert proxy.a == 1


# Check that changing the caller's aliases dict after construction doesn't change the proxy
def test_proxy_aliases_not_shared_with_caller():
    aliases = {"b": "a"}
    proxy = CategoryProxy({"a": 1, "c": 2}, aliases)
    aliases["b"] = "c"
    assert proxy.b == 1


# Check that the props mapping can't be used to change a value behind the proxy's back
def test_proxy_props_is_read_only():
    proxy = CategoryProxy({"a": 1})
    try:
        proxy.props["a"] = 2
    except TypeError:
        pass
    assert proxy.a == 1


# Check that the aliases mapping can't be used to change an alias behind the proxy's back
def test_proxy_aliases_is_read_only():
    proxy = CategoryProxy({"a": 1, "c": 2}, {"b": "a"})
    try:
        proxy.aliases["b"] = "c"
    except TypeError:
        pass
    assert proxy.b == 1

# endregion


# region CategoryProxy Get Attribute

# Check that we can get a property by name
def test_proxy_getattr_by_name():
    assert CategoryProxy({"a": 1}).a == 1


# Check that we can get a property by alias
def test_proxy_getattr_by_alias():
    assert CategoryProxy({"a": 1}, {"b": "a"}).b == 1


# Check that a missing property raises an error
def test_proxy_getattr_missing_raises():
    with pytest.raises(AttributeError):
        CategoryProxy({"a": 1}).missing


# Check that we raise an error when a private property is accessed
def test_proxy_getattr_private_raises():
    with pytest.raises(AttributeError):
        CategoryProxy({"_a": 1})._a


# endregion


# region CategoryProxy Set Attribute

# Check that we raise an error when you try to set a property by name
def test_proxy_setattr_public_blocked():
    proxy = CategoryProxy({"a": 1})
    with pytest.raises(AttributeError):
        proxy.a = 2

# Check that private attributes can still be set (needed for __init__ and unpickling)
def test_proxy_setattr_private_allowed():
    proxy = CategoryProxy({"a": 1})
    proxy._extra = 2
    assert proxy._extra == 2

# endregion


# region CategoryProxy Serialization

# Check that a proxy survives pickling
def test_proxy_pickle():
    proxy = pickle.loads(pickle.dumps(CategoryProxy({"a": 1}, {"b": "a"})))
    assert proxy.a == 1 and proxy.b == 1


# endregion


# region SettingsManager Cast Overrides

# Check that _cast_override returns the same value / type it was passed without changing an already correct type
def test_cast_same_type():
    prop = SettingsProperty("name", "cat", 1, int)
    assert SettingsManager._cast_override(prop, 5) == 5


# Check that we can cast a string to a bool
@pytest.mark.parametrize("input, expected", [
    ("yes", True), ("TRUE", True), ("t", True), ("y", True), ("1", True),
    ("no", False), ("False", False), ("f", False), ("n", False), ("0", False),
])
def test_cast_str_to_bool(input, expected):
    prop = SettingsProperty("name", "cat", False, bool)
    assert SettingsManager._cast_override(prop, input) is expected


# Check that we raise an error if the string is not a valid bool
def test_cast_bad_str_to_bool_raises():
    prop = SettingsProperty("name", "cat", False, bool)
    with pytest.raises(TypeError):
        SettingsManager._cast_override(prop, "maybe")


# Check that we can cast a string to an int
def test_cast_str_to_int():
    prop = SettingsProperty("name", "cat", 1, int)
    assert SettingsManager._cast_override(prop, "7") == 7


# Check that we warn if a float (passed as a string) is cast to an int
def test_cast_float_str_to_int_warns():
    prop = SettingsProperty("name", "cat", 1, int)
    with pytest.warns(UserWarning):
        assert SettingsManager._cast_override(prop, "7.9") == 7


# Check that we raise an error if the string is not a valid int
def test_cast_bad_str_to_int_raises():
    prop = SettingsProperty("name", "cat", 1, int)
    with pytest.raises(TypeError):
        SettingsManager._cast_override(prop, "seven")


# Check that we can cast a string to a float
def test_cast_str_to_float():
    prop = SettingsProperty("name", "cat", 1.0, float)
    assert SettingsManager._cast_override(prop, "1e9") == 1e9


# Check that we raise an error if the string is not a valid float
def test_cast_bad_str_to_float_raises():
    prop = SettingsProperty("name", "cat", 1.0, float)
    with pytest.raises(TypeError):
        SettingsManager._cast_override(prop, "big")


# Check that we warn if a float is cast to an int
def test_cast_float_to_int_warns():
    prop = SettingsProperty("name", "cat", 1, int)
    with pytest.warns(UserWarning):
        assert SettingsManager._cast_override(prop, 7.9) == 7


# check that we can cast an int to a float
def test_cast_int_to_float():
    prop = SettingsProperty("name", "cat", 1.0, float)
    result = SettingsManager._cast_override(prop, 3)
    assert result == 3.0 and type(result) is float


# Check that we can cast a numpy type to a built-in type
@pytest.mark.parametrize("prop_type, default_value, value", [
    (bool, False, np.bool_(False)),
    (float, 1.234, np.float64(1.234)),
    (int, 42, np.int64(42)),
])
def test_cast_numpy_to_builtin(prop_type, default_value, value):
    prop = SettingsProperty("name", "cat", default_value, prop_type)
    result = SettingsManager._cast_override(prop, value)
    assert result == value and type(result) is prop_type


# Check that 'None' is rejected for casting
def test_cast_none_rejected_for_required():
    prop = SettingsProperty("name", "cat", 1, int)
    with pytest.raises(TypeError):
        SettingsManager._cast_override(prop, None)


# Check that we raise an error for unsupported types
def test_cast_unsupported_type_raises():
    prop = SettingsProperty("name", "cat", 1, int)
    with pytest.raises(TypeError):
        SettingsManager._cast_override(prop, [1])


# Check that None is accepted for an optional property
def test_cast_none_allowed_for_optional():
    prop = OptionalSettingsProperty("name", "cat", "", str)
    assert SettingsManager._cast_override(prop, None) is None

# endregion


# region SettingsManager Initialization

# Check if an alias can be used to set values during initialization
def test_init_override_by_alias_name_get():
    assert SettingsManager({"mean_harden_energy_delta": 42}).mu_circ_encounter == 42


# Check if an alias can be used to set read values after initialization
def test_init_override_by_name_alias_get():
    assert SettingsManager({"mu_circ_encounter": 42}).mean_harden_energy_delta == 42


# Check if an alias override is properly cast to the correct type
def test_init_override_by_alias_cast():
    assert isinstance(SettingsManager({"mean_harden_energy_delta": "42"}).mu_circ_encounter, float)


# Check that passing the name and alias for the same properity raises an error
def test_init_override_by_name_and_alias_raises():
    with pytest.raises(ValueError):
        SettingsManager({"mu_circ_encounter": 42, "mean_harden_energy_delta": 43})


# Check that we can initialize a SettingsManager with no overrides
def test_default_settings():
    live = SettingsManager()
    for prop in DEFAULT_SETTINGS:
        assert getattr(live, prop.name) == prop.value, \
            f"{prop.name} has value {getattr(live, prop.name)}, " \
            f"but should be default ({prop.value})!"


# Check that we can pass overrides by name
def test_init_override_by_name():
    assert SettingsManager({"seed": 7}).seed == 7


# Check that overrided values are correctly cast
def test_init_override_is_cast():
    assert SettingsManager({"seed": "7"}).seed == 7


# Check that unknown keys raise an error
def test_init_unknown_key_raises():
    with pytest.raises(ValueError):
        SettingsManager({"definitely_not_a_setting": 1})


# Check that rg is set after init
def test_init_sets_r_g():
    assert SettingsManager().r_g_in_meters > 0


# Check that giving a setting by both name and alias raises and error
def test_init_name_and_alias_both_given_raises():
    with pytest.raises(ValueError):
        SettingsManager({"mu_circ_encounter": 0.5, "mean_harden_energy_delta": 0.6})


# Check that an alias override is cast to the correct type
def test_init_alias_override_is_cast():
    assert SettingsManager({"mean_harden_energy_delta": "0.5"}).mu_circ_encounter == 0.5


# Check that CLI-only args are accepted and ignored args and not stored
@pytest.mark.parametrize("key", sm.IGNORE_ARGS)
def test_init_ignored_key_accepted(key):
    assert key not in SettingsManager({key: None}).flat_settings


# Check that overriding a static setting keeps the default
@pytest.mark.parametrize("static_setting", [p for p in DEFAULT_SETTINGS if isinstance(p, StaticSettingsProperty) and p.prop_type is str])
def test_init_static_override_keeps_default(static_setting):
    with pytest.warns(UserWarning):
        live = SettingsManager({static_setting.name: static_setting.value + "_changed"})
        assert getattr(live, static_setting.name) == static_setting.value


# Check that overriding a static setting with a different value warns
@pytest.mark.parametrize("static_setting", [p for p in DEFAULT_SETTINGS if isinstance(p, StaticSettingsProperty) and p.prop_type is str])
def test_init_static_override_warns(static_setting):
    with pytest.warns(UserWarning):
        SettingsManager({static_setting.name: static_setting.value + "_changed"})


# Check that None is accepted for an optional setting at init
def test_init_optional_none():
    assert SettingsManager({"settings_file": None}).settings_file is None


# Check that r_g reflects an overridden smbh_mass
def test_init_r_g_uses_override():
    assert SettingsManager({"smbh_mass": 2e8}).r_g_in_meters != SettingsManager().r_g_in_meters

# endregion


# region SettingsManager Get Attribute

# Check that a category proxy is returned from flat access
def test_getattr_category_returns_proxy():
    assert isinstance(SettingsManager().sim, CategoryProxy)


# Check that fla access gets the same values as category access
def test_getattr_flat_matches_category():
    live = SettingsManager()
    for prop in DEFAULT_SETTINGS:
        assert getattr(live, prop.name) == getattr(getattr(live, prop.category), prop.name)


# Check that a missing property raises an error
def test_getattr_missing_raises_attributeerror():
    with pytest.raises(AttributeError):
        SettingsManager().nope


# Check that hasattr returns False for a missing property
def test_getattr_missing_works_with_hasattr():
    assert not hasattr(SettingsManager(), "nope")


# Check that a default alias can be read through flat access
def test_getattr_flat_by_alias():
    live = SettingsManager()
    assert live.mean_harden_energy_delta == live.mu_circ_encounter


# Check that a default alias can be read through category access without being overridden
def test_getattr_category_by_alias():
    live = SettingsManager()
    assert live.bh.mean_harden_energy_delta == live.bh.mu_circ_encounter


# Check that a missing private attribute raises AttributeError (deepcopy/pickle rely on this)
def test_getattr_private_raises_error():
    with pytest.raises(AttributeError):
        SettingsManager()._super_private_and_illegal_to_get

# endregion


# region SettingsManager set_preprocessing

# Check that changing a settings after init works with flat access
def test_set_preprocessing_updates_flat():
    live = SettingsManager()
    assert live.seed != 42 # Just check so we don't accidentally have a invalid test if the seed default changes to 42

    live.set_preprocessing("seed", 42)
    assert live.seed == 42


# Check that changing a settings after init works with category access
def test_set_preprocessing_updates_category():
    live = SettingsManager()
    assert live.sim.seed != 42

    live.set_preprocessing("seed", 7)
    assert live.sim.seed == 7


# Check that changing a settings after init casts the value to the correct type
def test_set_preprocessing_casts():
    live = SettingsManager()
    live.set_preprocessing("seed", "7")
    assert live.seed == 7


# Check that the rg updates when smbh_mass changes
def test_set_preprocessing_smbh_mass_updates_r_g():
    live = SettingsManager()
    r_g = live.r_g_in_meters
    live.set_preprocessing("smbh_mass", 2 * live.smbh_mass)
    assert live.r_g_in_meters != r_g


# Check that we raise an error when the key doesn't exist
def test_set_preprocessing_unknown_raises():
    with pytest.raises(ValueError):
        SettingsManager().set_preprocessing("nope", 1)


# Check that writing through a category's props can't desync it from the flat settings
def test_category_props_cannot_bypass_set_preprocessing():
    live = SettingsManager()
    seed = live.seed

    try:
        live.sim.props["seed"] = seed + 1
    except TypeError:
        pass

    assert live.sim.seed == live.seed == seed


# Check that a setting can be changed through its alias
def test_set_preprocessing_by_alias():
    live = SettingsManager()
    live.set_preprocessing("mean_harden_energy_delta", 0.5)
    assert live.mu_circ_encounter == 0.5


# Check that changing a setting is accessible through its alias
def test_set_preprocessing_visible_through_alias():
    live = SettingsManager()
    live.set_preprocessing("mu_circ_encounter", 0.5)
    assert live.mean_harden_energy_delta == 0.5


# Check that static settings can't be changed
@pytest.mark.parametrize("static_setting", [p for p in DEFAULT_SETTINGS if isinstance(p, StaticSettingsProperty) and p.prop_type is str])
def test_set_preprocessing_static_raises(static_setting):
    with pytest.raises(TypeError):
        SettingsManager().set_preprocessing(static_setting.name, static_setting.value)


# Check that a bad value raises and leaves the old value in place
def test_set_preprocessing_bad_value_keeps_old():
    live = SettingsManager()
    old = live.seed

    with pytest.raises(TypeError):
        live.set_preprocessing("seed", "forty two")

    assert live.seed == old

# endregion


# region SettingsManager Equal

# Check that two default initialized setting managers are equal
def test_eq_same_settings():
    assert SettingsManager() == SettingsManager()


# Check that setting managers with different settings are not equal
def test_eq_different_settings():
    assert SettingsManager() != SettingsManager({"seed": 42})


# Check that SettingsManager doesn't return true when compared to a different type
def test_eq_other_type():
    assert SettingsManager() != {}


# Check that adding a custom category makes two managers unequal
def test_eq_custom_category_counts():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1})
    assert live != SettingsManager()

# endregion


# region SettingsManager Copy Functions

# Check that copying with a replacement dictionary gives the correct value in the copy
def test_copy_replace():
    assert SettingsManager().copy({"seed": 42}).seed == 42

# Check that copying also copies the custom categories
def test_copy_with_custom_category():
    live = SettingsManager()
    live.add_custom_category("mine", {"a": 42})
    new = live.copy()

    assert new.mine.a == 42
    assert live == new


# Check that copy accepts an alias in replace
def test_copy_replace_by_alias():
    assert SettingsManager().copy({"mean_harden_energy_delta": 0.5}).mu_circ_encounter == 0.5


# Check that copy with an unknown replacement key raises an error
def test_copy_replace_unknown_raises():
    with pytest.raises(ValueError):
        SettingsManager().copy({"no_such_setting": 1})


# Check that copy updates r_g when smbh_mass is replaced
def test_copy_replace_smbh_mass_updates_r_g():
    live = SettingsManager()
    assert live.copy({"smbh_mass": 2 * live.smbh_mass}).r_g_in_meters != live.r_g_in_meters


# Check that copy keeps custom category aliases
def test_copy_custom_category_keeps_aliases():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1}, {"applepen": "pen"})
    assert live.copy().pineapple.applepen == 1


# Check that a copy doesn't share its custom category storage with the original
def test_copy_custom_category_not_shared():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1}, {"applepen": "pen"})
    new = live.copy()

    assert new.pineapple.props is not live.pineapple.props
    assert new.pineapple.aliases is not live.pineapple.aliases

# endregion


# region SettingsManager Add Custom Category

# Check that we can add a custom cat and access it
def test_add_custom_category_access():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1})
    assert live.pineapple.pen == 1


# Check that changing the caller's dicts after adding a custom category doesn't change the settings
def test_add_custom_category_not_shared_with_caller():
    props = {"pen": 1}
    aliases = {"applepen": "pen"}

    live = SettingsManager()
    live.add_custom_category("pineapple", props, aliases)

    props["pen"] = 2
    aliases["pineapplepen"] = "pen"

    assert live.pineapple.pen == 1
    with pytest.raises(AttributeError):
        live.pineapple.pineapplepen


# Check that we can add a custom cat with aliases and access with the alias and the original name
def test_add_custom_category_alias_access():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1}, {"applepen": "pen"})
    assert live.pineapple.applepen == 1
    assert live.pineapple.pen == 1


# Check that we raise an error when the category name already exists
def test_add_custom_category_existing_category_raises():
    with pytest.raises(AttributeError):
        SettingsManager().add_custom_category("sim", {"a": 1})


# Check that we raise an error when the category name is a default key
def test_add_custom_category_setting_name_raises():
    with pytest.raises(AttributeError):
        SettingsManager().add_custom_category("seed", {"a": 1})


# Check that we raise an error if the alias already exists as a default key
def test_add_custom_category_alias_clash_raises():
    with pytest.raises(AttributeError):
        SettingsManager().add_custom_category("pineapple", {"pen": 1}, {"seed": "pen"})


# Check that we raise an error when a category is added with a name that already is a category
def test_add_custom_category_existing_category_name():
    live = SettingsManager()

    live.add_custom_category("pineapple", {"pen": 1})

    with pytest.raises(AttributeError):
        live.add_custom_category("pineapple", {"applepen": 2})


# Check that a custom category's property doesn't overwrite a default property with the same name
def test_add_custom_category_does_not_shadow_default():
    live = SettingsManager()
    with pytest.raises(AttributeError):
        live.add_custom_category("pineapple", {"seed": 42})


# Check that a category can't have the same name as a default alias
def test_add_custom_category_name_is_default_alias_raises():
    with pytest.raises(AttributeError):
        SettingsManager().add_custom_category("mean_harden_energy_delta", {"a": 1})


# Check that a custom prop can't have the same name as a default alias
def test_add_custom_category_prop_is_default_alias_raises():
    with pytest.raises(AttributeError):
        SettingsManager().add_custom_category("pineapple", {"mean_harden_energy_delta": 1})


# Check that two custom categories can't share aliases
def test_add_custom_category_duplicate_custom_alias_raises():
    live = SettingsManager()
    live.add_custom_category("pineapple", {"pen": 1}, {"applepen": "pen"})
    with pytest.raises((AttributeError, ValueError)):
        live.add_custom_category("apple", {"pencil": 1}, {"applepen": "pencil"})


# Check that a failed add leaves the manager unchanged
def test_add_custom_category_fails_clean():
    live = SettingsManager()

    live.add_custom_category("pineapple", {"pen": 1}, {"applepen": "pen"})

    with pytest.raises((AttributeError, ValueError)):
        live.add_custom_category("apple", {"pencil": 1}, {"applepen": "pencil"})

    assert "apple" not in live.categories

    live.set_preprocessing("seed", 7)


# Check that set_preprocessing on a custom prop fails cleanly (custom props aren't preprocessed)
def test_set_preprocessing_custom_prop_raises():
    live = SettingsManager()

    live.add_custom_category("pineapple", {"pen": 1})

    with pytest.raises(ValueError):
        live.set_preprocessing("pen", 2)

# endregion


# region SettingsManager Serialization

# Check that passing a static setting at its default value doesn't warn
def test_atomic_settings():
    # Define some settings
    live = SettingsManager()
    # Loop
    for key, value in live.flat_settings.items():
        assert deepcopy(value) == value


def test_pickle():
    # Define some settings
    live = SettingsManager()
    # Pickle them
    jar = pickle.loads(pickle.dumps(live))
    # Check they are equal
    assert live == jar
    # Check they are not the same
    assert live is not jar


def test_copy():
    # Define some settings
    live = SettingsManager()
    # Make a copy
    copy = live.copy()
    # Check they are equal
    assert live == copy
    # Check they are not the same
    assert live is not copy
    # Make another copy
    copy = live.copy({"seed": 42})
    # Check that they are not equal
    assert live != copy

# endregion


def test_process_pool():
    # Define some settings
    live = SettingsManager()
    # Initialize monkey pool
    monkeys = np.empty((N_PROC,), dtype=bool)
    # Try tqdm
    with tqdm.tqdm(
            total=N_PROC, desc=f"{N_PROC} Litte monkeys jumping on the bed"
    ) as pbar:
        # Recite the incantation
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROC) as executor:
            # Submit processes
            future_to_monkey = {executor.submit(
                monkey_hit_head,
                live.copy({"seed": live.seed - int(monkey)})
            ): monkey for monkey in np.arange(N_PROC)}
            # Catch processes
            for future in concurrent.futures.as_completed(future_to_monkey):
                # Get the index
                monkey_see = future_to_monkey[future]
                # Assign the value
                monkeys[monkey_see] = future.result()  # Monkey do
                # Update tqdm as processes complete
                pbar.update(1)
    # Assert Monkeys
    print(f"{monkeys.sum()} fell off and bumped their head!")
    assert monkeys.sum() == 8, "Default seed has probably changed."


######## Main ########
def main():
    pytest.main([__file__])


######## Execution ########
if __name__ == "__main__":
    main()
