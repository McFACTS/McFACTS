"""
Module for hardening the binary via gas.
"""
import astropy.units as u
import astropy.constants as const
import numpy as np
from numpy.random import Generator

import mcfacts.modules.gw
import mcfacts.utilities.checks
import mcfacts.utilities.peters
import mcfacts.utilities.unit_conversion
from mcfacts.inputs.settings_manager import AGNDisk, SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNBlackHoleArray
from mcfacts.objects.timeline import TimelineActor
from mcfacts.utilities import peters, checks, unit_conversion

def bin_harden_baruteau_optimized( bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging, bin_time_merged, smbh_mass, timestep_duration_yr, time_gw_normalization, time_passed, r_g_in_meters):
    return baruteau_helper(
        bin_mass_1,
        bin_mass_2,
        bin_sep,
        bin_ecc,
        bin_time_to_merger_gw,
        bin_flag_merging,
        bin_time_merged,
        smbh_mass,
        timestep_duration_yr,
        time_passed
    )


def baruteau_drag(mass_1, mass_2, bin_sep, smbh_mass, timestep_duration_yr):
    binary_mass = mass_1 + mass_2
    bin_period = 0.32 * np.power(bin_sep, 1.5) * np.power(smbh_mass / 1.e8, 1.5) * np.power(
        binary_mass / 10.0, -0.5)

    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    return bin_sep * (0.5 ** scaled_num_orbits)


def stahler_drag(mass_1, mass_2, bin_sep, orb_a, disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters):
    q = np.minimum(mass_1 / mass_2, mass_2 / mass_1)

    total_mass = ((mass_1 + mass_2) * const.M_sun).si

    scaling_constant = (15 / (35 * np.pi))
    ratio_component = (((1 + q) ** 2) / q)
    gas_component = (((disk_sound_speed(orb_a) * u.meter/u.second) ** 5) / (disk_density(orb_a) * (u.kg / u.m ** 3)))
    mass_component = 1 / ((const.G ** 3) * (total_mass ** 2))

    sep_unit = unit_conversion.si_from_r_g(smbh_mass, bin_sep, r_g_defined=r_g_in_meters)

    coalescence_time = sep_unit * (scaling_constant * ratio_component * gas_component * mass_component)

    timestep_units = (timestep_duration_yr * u.year).si

    new_bin_sep = bin_sep * (1 - (timestep_units / coalescence_time))

    contact_condition = (unit_conversion.r_schwarzschild_of_m(mass_1) +
                         unit_conversion.r_schwarzschild_of_m(mass_2))
    contact_condition = unit_conversion.r_g_from_units(smbh_mass, contact_condition).value

    new_bin_sep[new_bin_sep < contact_condition] = contact_condition[new_bin_sep < contact_condition]

    return new_bin_sep


def primary_drag_force_components_base(mach_number):
    radial_component = 0.3 * (mach_number ** 2)
    azimuthal_component = np.log(10 / ((0.11 * mach_number) + 1.65))

    if mach_number < 6.2:
        radial_component = 4 + (mach_number ** 2) * (np.e ** -(mach_number - 7)) * np.sin(((mach_number - 4.4) / 4)) / 5

    if 1.1 <= mach_number < 4.4:
        radial_component = 0.5 * np.log(9.33 * (mach_number ** 2) * (mach_number ** 2 - 0.95))
    if 1.0 <= mach_number < 4.4:
        azimuthal_component = np.log(3300 * ((mach_number - 0.71) ** 5.72) * (mach_number ** -9.58))

    if mach_number < 1.1:
        radial_component = (mach_number ** 2) * (10 ** ((3.51 * mach_number) - 4.22))
    if mach_number < 1.0:
        azimuthal_component = 0.7706 * np.log(
            (1 + mach_number) / (1.0004 - 0.9185 * mach_number)) - 1.4703 * mach_number

    if mach_number <= 0.0523352:
        azimuthal_component = 0.0

    return radial_component, azimuthal_component


def secondary_drag_force_components_base(mach_number):
    radial_component = 0.56 - (0.027 * (mach_number + ((mach_number - 6) ** -1)))
    azimuthal_component = -0.13 + 0.07 * np.arctan((5 * mach_number) - 15)

    if 2.97 <= mach_number < 6.2:
        radial_component = 0.76 - (0.08 * (mach_number + ((mach_number - 2.76) ** -1)))

    if mach_number < 2.97:
        radial_component = 0.5 - (0.43 * (1 - np.cosh(2.2 * mach_number) ** -0.36))
        azimuthal_component = -0.022 * (10 - mach_number) * np.tanh(3 * mach_number / 2)

    radial_component *= mach_number ** 2
    azimuthal_component *= mach_number ** 2

    return radial_component, azimuthal_component


def drag_forces(mass, velocity, density, sound_speed, func_drag_force_components):
    mach_number = velocity / sound_speed

    drag_force_components = func_drag_force_components(mach_number.value)
    force_component = (4 * np.pi * density) * (((const.G * mass) / velocity) ** 2)

    return (-force_component * drag_force_components[0]), (-force_component * drag_force_components[1])


def analytical_drag(mass_1, mass_2, bin_sep, bin_orb_a, flag_merging, disk_sound_speed, disk_density, timestep_length, smbh_mass):
    # Return early if there are no binaries
    if len(mass_1) == 0:
        return bin_sep

    # Define the drag force methods to allow for numpy array operations
    primary_drag_force_components = np.vectorize(primary_drag_force_components_base)
    secondary_drag_force_components = np.vectorize(secondary_drag_force_components_base)

    # Mass Ratio
    q = np.minimum(mass_1, mass_2) / np.maximum(mass_1, mass_2)

    # Convert binary separation to si units
    si_unit_bin_sep = unit_conversion.si_from_r_g(smbh_mass, bin_sep)

    # Add units to mass and calc total mass
    unit_mass_1 = (mass_1 * u.M_sun).si
    unit_mass_2 = (mass_2 * u.M_sun).si
    total_mass = unit_mass_1 + unit_mass_2

    # Get the local sound speed and density at the binary's location in the disk
    sound_speed = disk_sound_speed(bin_orb_a) * (u.m / u.s)
    density = disk_density(bin_orb_a) * (u.kg / u.m ** 3)

    # Size of sub steps (in years) to take within the simulation timestep
    # Need to take substeps since analytical function can run away under large timesteps
    sub_step_size = 100 * u.yr
    sub_steps = (timestep_length * u.yr) / sub_step_size

    # Merging flag might be stale, so lets run a contact check just incase
    _, flag_merging = checks.bin_contact_check(
        mass_1,
        mass_2,
        bin_sep,
        flag_merging,
        smbh_mass,
    )

    # Loop over the number of sub-timesteps
    # TODO: Possible implementation for system handling sub-timesteps for multiple modules
    for n in range(int(sub_steps.value)):
        # Turn our merging flag into a mask
        not_merging = flag_merging >= 0

        # Find the semi-major axis of each binary component to the center of mass based on mass ratio
        sep_1 = si_unit_bin_sep[not_merging] / ((1 / q[not_merging]) + 1)
        sep_2 = si_unit_bin_sep[not_merging] / (q[not_merging] + 1)

        # Find the Keplerian orbital velocity for w.r.t. the center of mass for each binary component
        orb_vel_1 = np.sqrt((const.G * total_mass[not_merging] / sep_1))
        orb_vel_2 = np.sqrt((const.G * total_mass[not_merging] / sep_2))

        # Using Kim+Kim+Sánchez-Salcedo semi-analytical model for double peturbers,
        # find the drag force acting on each component due to the tails of the component and the companion
        prime_df_1 = drag_forces(unit_mass_1[not_merging], orb_vel_1, density[not_merging], sound_speed[not_merging], primary_drag_force_components)
        second_df_1 = drag_forces(unit_mass_2[not_merging], orb_vel_2, density[not_merging], sound_speed[not_merging], secondary_drag_force_components)

        prime_df_2 = drag_forces(unit_mass_2[not_merging], orb_vel_2, density[not_merging], sound_speed[not_merging], primary_drag_force_components)
        second_df_2 = drag_forces(unit_mass_1[not_merging], orb_vel_1, density[not_merging], sound_speed[not_merging], secondary_drag_force_components)

        # Find the acceleration on each component using the force in the phi direction
        accel_phi_1 = (prime_df_1[1] + second_df_1[1]) / unit_mass_1[not_merging]
        accel_phi_2 = (prime_df_2[1] + second_df_2[1]) / unit_mass_2[not_merging]

        # Calculate a new orbital velocity for each component, change of velocity over substep
        new_orb_vel_1 = orb_vel_1 + (accel_phi_1 * (sub_step_size.to(u.s)))
        new_orb_vel_2 = orb_vel_2 + (accel_phi_2 * (sub_step_size.to(u.s)))

        # Find the new separations corresponding to the new orbital velocities
        new_sep_1 = ((const.G * total_mass[not_merging]) / (new_orb_vel_1 ** 2))
        new_sep_2 = ((const.G * total_mass[not_merging]) / (new_orb_vel_2 ** 2))

        # Add the component separations back together
        si_unit_bin_sep[not_merging] = new_sep_1 + new_sep_2

        # Check if any binaries would merge, if so update our flag_merging array so they don't get evolved in the next substep
        merged_bin_sep, flag_merging = checks.bin_contact_check(
            mass_1,
            mass_2,
            si_unit_bin_sep.value,
            flag_merging,
            smbh_mass,
        )

    return unit_conversion.r_g_from_units(smbh_mass, si_unit_bin_sep).value


def gas_hardening_no_stalling(mass_1, mass_2, bin_sep, flag_merging, smbh_mass, gas_hardening_prescription, orb_a, disk_sound_speed, disk_density, timestep_duration_yr, r_g_in_meters):
    flag_not_merging = np.array([(flag_merging[i] >= 0) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    elif gas_hardening_prescription == "analytical":
        calc_bin_sep = analytical_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], orb_a[flag_not_merging], flag_merging[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass)
    else:
        assert False, "Incorrect gas hardening prescription specified... Available values: (baruteau, stahler, analytical)"

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


def gas_hardening_variable_stalling(mass_1, mass_2, bin_sep, bin_orb_a, disk_sound_speed, flag_merging, smbh_mass, gas_hardening_prescription, disk_density, timestep_duration_yr, r_g_in_meters):
    rg_scale = (const.G * smbh_mass * const.M_sun / const.c ** 2).value
    bin_orb_velocity = np.sqrt((const.G.value * ((mass_1 + mass_2) * const.M_sun).si.value) / (bin_sep * rg_scale))
    sound_speed = disk_sound_speed(bin_orb_a)

    effective_stalling_separation = np.array([(sep if vel >= s_speed else 0) for vel, sep, s_speed in zip(bin_orb_velocity, bin_sep, sound_speed)])
    flag_not_merging = np.array([(flag_merging[i] >= 0 and bin_sep[i] > effective_stalling_separation[i]) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], bin_orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    else:
        assert False, "Incorrect gas hardening prescription specified... Available values: (baruteau, stahler)"

    calc_bin_sep = np.maximum(calc_bin_sep, effective_stalling_separation[flag_not_merging])

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


def gas_hardening_fixed_stalling(mass_1, mass_2, bin_sep, flag_merging, smbh_mass, stalling_separation, gas_hardening_prescription, orb_a, disk_sound_speed, disk_density,  timestep_duration_yr, r_g_in_meters):
    flag_not_merging = np.array([(flag_merging[i] >= 0 and bin_sep[i] > stalling_separation) for i in range(len(mass_1))], dtype=bool)

    if gas_hardening_prescription == "baruteau":
        calc_bin_sep = baruteau_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], smbh_mass, timestep_duration_yr)
    elif gas_hardening_prescription == "stahler":
        calc_bin_sep = stahler_drag(mass_1[flag_not_merging], mass_2[flag_not_merging], bin_sep[flag_not_merging], orb_a[flag_not_merging], disk_sound_speed, disk_density, timestep_duration_yr, smbh_mass, r_g_in_meters)
    else:
        assert False, "Incorrect gas hardening prescription specified... Available values: (baruteau, stahler)"

    calc_bin_sep[calc_bin_sep < stalling_separation] = stalling_separation

    new_bin_sep = np.zeros(len(mass_1))
    new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
    new_bin_sep[flag_not_merging] = calc_bin_sep

    return new_bin_sep


class BinaryBlackHoleGasHardening(TimelineActor):
    def __init__(self, name: str = None, settings: SettingsManager = None, reality_merge_checks: bool = False):
        super().__init__("Binary Black Hole Gas Hardening" if name is None else name, settings)

        self.reality_merge_checks = reality_merge_checks

    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        sm = self.settings

        if sm.bbh_array_name not in filing_cabinet:
            return

        blackholes_binary = filing_cabinet.get_array(sm.bbh_array_name, AGNBlackHoleArray)

        if sm.stalling_separation > 0:
            blackholes_binary.bin_sep = gas_hardening_fixed_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.stalling_separation,
                sm.gas_hardening_prescription,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )
        elif sm.stalling_separation == 0:
            blackholes_binary.bin_sep = gas_hardening_no_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.gas_hardening_prescription,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )
        elif sm.stalling_separation == -1:
            blackholes_binary.bin_sep = gas_hardening_variable_stalling(
                blackholes_binary.mass,
                blackholes_binary.mass_2,
                blackholes_binary.bin_sep,
                blackholes_binary.orb_a,
                agn_disk.disk_sound_speed,
                blackholes_binary.flag_merging,
                sm.smbh_mass,
                sm.gas_hardening_prescription,
                agn_disk.disk_density,
                timestep_length,
                sm.r_g_in_meters
            )


        if not self.reality_merge_checks:
            return

        checks.binary_reality_check(sm, filing_cabinet, self.log)
        checks.flag_binary_mergers(sm, filing_cabinet)
