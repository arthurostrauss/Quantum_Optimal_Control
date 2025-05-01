import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Literal, List, Union

from qm.qua import fixed

from .parameter import Parameter
from .parameter_table import ParameterTable


def get_cpp_type(parameter: Parameter):
    """Returns the parameter's QUA variable type in C++ syntax."""
    if parameter.type == int:
        return "int"
    elif parameter.type == fixed:
        return "double"
    else:  # Bool
        return "bool"


def make_cpp_packet_definition(
    direction: Literal["Incoming", "Outgoing"],
    param_table: Union[ParameterTable, Parameter],
) -> str:
    """Constructs the C++ code defining the QM packet structure."""
    packet_definition = f"QM_DECLARE_PACKET(Internal{direction}Packet, "

    parameters = (
        param_table.parameters if isinstance(param_table, ParameterTable) else [param_table]
    )
    for i, parameter in enumerate(parameters):
        packet_definition += (
            f"({parameter.name}, {get_cpp_type(parameter)}, "
            f"{parameter.length if parameter.is_array else 1})"
        )
        if i != len(parameters) - 1:
            packet_definition += ", "
    packet_definition += ");"

    return packet_definition


def make_h_packet_definition(
    direction: Literal["Incoming", "Outgoing"],
    param_table: Union[ParameterTable, Parameter],
) -> str:
    """Constructs the C++ header code defining the QM packet structure."""
    struct_name = f"{direction}Packet"
    packet_definition = f"struct {struct_name} " + "{ "

    parameters = (
        param_table.parameters if isinstance(param_table, ParameterTable) else [param_table]
    )
    for parameter in parameters:
        packet_definition += f"std::vector<{get_cpp_type(parameter)}> {parameter.name}{{}}; "

    if direction == "Incoming":
        pass
    elif direction == "Outgoing":
        packet_definition += f"{struct_name}("
        for i, parameter in enumerate(parameters):
            if parameter.is_array:
                for j in range(parameter.length):
                    packet_definition += f"{get_cpp_type(parameter)} _{parameter.name}_{j}"
                    if not (i == len(parameters) - 1 and j == parameter.length - 1):
                        packet_definition += ", "
            else:
                packet_definition += f"{get_cpp_type(parameter)} _{parameter.name}"
                if i != len(parameters) - 1:
                    packet_definition += ", "

        packet_definition += ") { "

        for i, parameter in enumerate(parameters):
            if parameter.is_array:
                for j in range(parameter.length):
                    packet_definition += f"{parameter.name}.push_back(_{parameter.name}_{j}); "
            else:
                packet_definition += f"{parameter.name}.push_back(_{parameter.name}); "

        packet_definition += "} "

    packet_definition += "};"

    return packet_definition


def make_i_packet_definition(
    direction: Literal["Incoming", "Outgoing"],
    param_table: Union[ParameterTable, Parameter],
) -> str:
    """Constructs the C++ header code defining the QM packet structure."""
    struct_name = f"{direction}Packet"
    packet_definition = f"extern struct {struct_name} " + "{ "

    parameters = (
        param_table.parameters if isinstance(param_table, ParameterTable) else [param_table]
    )
    for parameter in parameters:
        packet_definition += f"std::vector<{get_cpp_type(parameter)}> {parameter.name}; "

    if direction == "Incoming":
        # Add default constructor signature
        packet_definition += f"{struct_name}(); "

    elif direction == "Outgoing":
        # Add custom constructor signature
        packet_definition += f"{struct_name}("
        for i, parameter in enumerate(parameters):
            if parameter.is_array:
                for j in range(parameter.length):
                    packet_definition += f"{get_cpp_type(parameter)} _{parameter.name}_{j}"
                    if not (i == len(parameters) - 1 and j == parameter.length - 1):
                        packet_definition += ", "
            else:
                packet_definition += f"{get_cpp_type(parameter)} _{parameter.name}"
                if i != len(parameters) - 1:
                    packet_definition += ", "
            packet_definition += f"{get_cpp_type(parameter)} _{parameter.name}[{parameter.length if parameter.is_array else 1}]"
            if i != len(parameters) - 1:
                packet_definition += ", "
        packet_definition += "); "

    packet_definition += "};"

    return packet_definition


def make_outgoing_packet_typecast(param_table: Union[ParameterTable, Parameter]) -> str:
    """const InternalOutgoingPacket out_packet{qm::Value<double, 10>(packet.data)};"""
    definition = "const InternalOutgoingPacket out_packet{ "
    parameters = (
        param_table.parameters if isinstance(param_table, ParameterTable) else [param_table]
    )

    for i, parameter in enumerate(parameters):
        definition += f"qm::Value<{get_cpp_type(parameter)},{parameter.length if parameter.is_array else 1}>(packet.{parameter.name})"
        if i != len(parameters) - 1:
            definition += ", "
    definition += "};"

    return definition


def make_ingoing_packet_typecast(param_table: Union[ParameterTable, Parameter]) -> str:
    """
    Constructs the C++ code to typecast the IncomingPacket into an InternalIncomingPacket.
    Args:
        param_table: The parameter table to generate the typecast for.

    Returns:
        The C++ code to typecast the IncomingPacket into an InternalIncomingPacket.
    """

    definition = "IncomingPacket incoming_packet; "
    parameters = (
        param_table.parameters if isinstance(param_table, ParameterTable) else [param_table]
    )

    for parameter in parameters:
        definition += f"for (int i = 0; i < {parameter.length if parameter.is_array else 1}; ++i) incoming_packet.{parameter.name}.push_back(packet.{parameter.name}[i]);"

    return definition


def file_checksum(path):
    """Returns the SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except FileNotFoundError:
        return None  # Handle missing files gracefully


def run_sed_and_check_for_changes(sed_args: List[str]):
    """Runs sed and checks if the file was modified."""
    file_path = sed_args[-1]
    before = file_checksum(file_path)
    subprocess.call(["sed", "-i", "-e"] + sed_args)
    after = file_checksum(file_path)
    return before is not None and before != after


def patch_opnic_wrapper(
    param_tables: List[Union[ParameterTable, Parameter]],
    path_to_opnic_dev="/home/dpoulos/aps_demo",
):
    """Modifies the source code of the OPNIC wrapper, appending new definitions only if they are not already present."""
    path_to_opnic_dev = Path(path_to_opnic_dev)
    paths = {
        "cpp": path_to_opnic_dev / "python-wrapper/wrapper/wrapper.cpp",
        "h": path_to_opnic_dev / "python-wrapper/wrapper/wrapper.h",
        "i": path_to_opnic_dev / "python-wrapper/wrapper/wrapper.i",
        "parent": path_to_opnic_dev / "python-wrapper/wrapper",
        "build_sh": path_to_opnic_dev / "python-wrapper/wrapper/build.sh",
    }
    replacements = []

    for param_table in param_tables:
        direction: Literal["Incoming", "Outgoing"] = (
            "Incoming" if param_table.direction == "INCOMING" else "Outgoing"
        )
        direction_ = "Inc" if direction == "Incoming" else "Out"
        cpp_packet = make_cpp_packet_definition(direction, param_table)
        h_packet = make_h_packet_definition(direction, param_table)
        i_packet = make_i_packet_definition(direction, param_table)
        typecast = (
            make_outgoing_packet_typecast(param_table)
            if direction == "Outgoing"
            else make_ingoing_packet_typecast(param_table)
        )

        replacements_ = [[f"s/QM_D.*{direction_}.*/{cpp_packet}/g", str(paths["cpp"])]]
        if direction == "Outgoing":
            replacements_.append([f"s/const Int.* out_packet.*/{typecast}/g", str(paths["cpp"])])
        else:
            replacements_.append(
                [f"s/IncomingPacket incoming_packet.*/{typecast}/g", str(paths["cpp"])]
            )

        replacements_.append([f"s/struct {direction_}.*/{h_packet}/g", str(paths["h"])])
        replacements_.append([f"s/.*struct {direction_}.*/{i_packet}/g", str(paths["i"])])
        replacements.extend(replacements_)

        modified_count = sum(run_sed_and_check_for_changes(args) for args in replacements)

        if modified_count > 0:
            logging.info(f"{modified_count} C++ files. Recompiling...")

            # Re-compile
            cwd = os.getcwd()
            os.chdir(paths["parent"])
            subprocess.call([f"{paths['build_sh']}"])
            os.chdir(cwd)

        else:
            logging.info(f"No modified C++ files. Skipping compilation...")
