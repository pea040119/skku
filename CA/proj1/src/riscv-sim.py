import struct
import sys

OPCODES = {
    0x37: "lui",
    0x17: "auipc",
    0x6F: "jal",
    0x67: "jalr",
    0x63: {0x0: "beq", 0x1: "bne", 0x4: "blt", 0x5: "bge", 0x6: "bltu", 0x7: "bgeu"},
    0x03: {0x0: "lb", 0x1: "lh", 0x2: "lw", 0x4: "lbu", 0x5: "lhu"},
    0x23: {0x0: "sb", 0x1: "sh", 0x2: "sw"},
    0x13: {0x0: "addi", 0x1: {0x00: "slli"}, 0x2: "slti", 0x3: "sltiu", 0x4: "xori", 0x5: {0x00: "srli", 0x20: "srai"}, 0x6: "ori", 0x7: "andi"},
    0x33: {0x0: {0x00: "add", 0x20: "sub"}, 0x1: "sll", 0x2: "slt", 0x3: "sltu", 0x4: "xor", 0x5: {0x00: "srl", 0x20: "sra"}, 0x6: "or", 0x7: "and"}
}

REGISTERS = ["x" + str(i) for i in range(32)]



def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def decode_instruction(inst):
    opcode = inst & 0x7F
    func_3 = (inst >> 12) & 0x7
    func_7 = (inst >> 25) & 0x7F
    rd = (inst >> 7) & 0x1F
    rs1 = (inst >> 15) & 0x1F
    rs2 = (inst >> 20) & 0x1F
    shamt = (inst >> 20) & 0x1F
    
    # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7:{hex(func_7)}")

    if opcode in OPCODES:
        if isinstance(OPCODES[opcode], dict):
            if func_3 in OPCODES[opcode]:
                if isinstance(OPCODES[opcode][func_3], dict):
                    if func_7 in OPCODES[opcode][func_3]:
                        instruction = OPCODES[opcode][func_3][func_7]
                    else:
                        return f"unknown instruction"
                else:
                    instruction = OPCODES[opcode][func_3]
            else:
                return f"unknown instruction"
        else:
            instruction = OPCODES[opcode]

        match opcode:
            case 0x33:
                return f"{instruction} {REGISTERS[rd]}, {REGISTERS[rs1]}, {REGISTERS[rs2]}"
            case 0x03:
                imm_se = sign_extend(inst >> 20, 12)
                return f"{instruction} {REGISTERS[rd]}, {imm_se}({REGISTERS[rs1]})"
            case 0x13:
                if func_3 in {0x1, 0x5}:
                    return f"{instruction} {REGISTERS[rd]}, {REGISTERS[rs1]}, {shamt}"
                else:
                    imm_se = sign_extend(inst >> 20, 12)
                    return f"{instruction} {REGISTERS[rd]}, {REGISTERS[rs1]}, {imm_se}"
            case 0x23:
                imm_se = sign_extend(((inst >> 7) & 0x1F) | ((inst >> 25) << 5), 12)
                return f"{instruction} {REGISTERS[rs2]}, {imm_se}({REGISTERS[rs1]})"
            case 0x63:
                imm_se = sign_extend(((inst >> 8) & 0xF) << 1 | ((inst >> 25) & 0x3F) << 5 | ((inst >> 7) & 0x1) << 11 | ((inst >> 31) & 0x1) << 12, 13)
                return f"{instruction} {REGISTERS[rs1]}, {REGISTERS[rs2]}, {imm_se}"
            case 0x37:
                imm_se = sign_extend(inst & 0xFFFFF000, 32)
                return f"{instruction} {REGISTERS[rd]}, {imm_se}"
            case 0x17:
                imm_se = inst & 0xFFFFF000
                return f"{instruction} {REGISTERS[rd]}, {imm_se}"
            case 0x6F:
                imm_se = sign_extend(((inst >> 31) & 0x1) << 20 | ((inst >> 12) & 0xFF) << 12 | ((inst >> 20) & 0x1) << 11 | ((inst >> 21) & 0x3FF) << 1, 21)
                return f"{instruction} {REGISTERS[rd]}, {imm_se}"
            case 0x67:
                imm_se = sign_extend(inst >> 20, 12)
                return f"{instruction} {REGISTERS[rd]}, {imm_se}({REGISTERS[rs1]})"
            case _:
                return f"unknown instruction"
    else:
        return f"unknown instruction"


def read_binary_file(filename):
    with open(filename, "rb") as f:
        binary_data = f.read()
        file_iter = []
        for i in range(0, len(binary_data), 4):
            inst = struct.unpack("<I", binary_data[i:i+4])[0]
            file_iter.append(inst)
        return file_iter


def riscv_sim(filename):
    file_iter = read_binary_file(filename)
    for i, inst in enumerate(file_iter):
        disassembled_inst = decode_instruction(inst)
        print(f"inst {i}: {inst:08x} {disassembled_inst}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[!] Error: Usage: python3 riscv-sim.py <binary_file>")
    else:
        riscv_sim(sys.argv[1])
