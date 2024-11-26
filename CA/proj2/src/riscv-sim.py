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



class RiscVSim:
    def __init__(self, filename, data):
        self.filename = filename
        self.registers = [0] * 32
        self.pc = 0
        self.data = data


    def sign_extend(self, value, bits):
        sign_bit = 1 << (bits - 1)
        return (value & (sign_bit - 1)) - (value & sign_bit)

    def normalize_32bit(self, value):
        # value &= 0xFFFFFFFF
        return value
    
    
    def __add(self, rs1, rs2):
        return self.normalize_32bit(rs1 + rs2)
    
    def __addi(self, rs1, imm):
        return self.normalize_32bit(rs1 + imm)
    
    def __sub(self, rs1, rs2):
        return self.normalize_32bit(rs1 - rs2)
    
    
    def __xor(self, rs1, rs2):
        return self.normalize_32bit(rs1 ^ rs2)
    
    def __or(self, rs1, rs2):
        return self.normalize_32bit(rs1 | rs2)
    
    def __and(self, rs1, rs2):
        return self.normalize_32bit(rs1 & rs2)
    
    def __xori(self, rs1, imm):
        return self.normalize_32bit(rs1 ^ imm)
    
    def __ori(self, rs1, imm):
        return self.normalize_32bit(rs1 | imm)
    
    def __andi(self, rs1, imm):
        return self.normalize_32bit(rs1 & imm)
    
    
    def __slli(self, rs1, shamt):
        rs1 &= 0xFFFFFFFF
        return self.normalize_32bit((rs1 << shamt) & 0xFFFFFFFF)

    def __srli(self, rs1, shamt):
        rs1 &= 0xFFFFFFFF
        return (rs1 >> shamt) & 0xFFFFFFFF

    def __srai(self, rs1, shamt):
        return self.normalize_32bit(rs1 >> shamt)

    def __sll(self, rs1, rs2):
        rs1 &= 0xFFFFFFFF
        rs2 &= 0x1F
        return self.normalize_32bit((rs1 << rs2) & 0xFFFFFFFF)

    def __srl(self, rs1, rs2):
        rs1 &= 0xFFFFFFFF
        rs2 &= 0x1F
        return (rs1 >> rs2) & 0xFFFFFFFF

    def __sra(self, rs1, rs2):
        rs2 &= 0x1F
        return self.normalize_32bit(rs1 >> rs2)
    
    
    def __slti(self, rs1, imm):
        return 1 if rs1 < imm else 0
    
    def __slt(self, rs1, rs2):
        return 1 if rs1 < rs2 else 0
    
    def __auipc(self, pc, imm):
        return (pc + imm)
    
    def __lui(self, imm20):
        return self.normalize_32bit(self.sign_extend(imm20, 20) << 12)
    
    
    def __jal(self, imm):
        pc = self.pc
        self.pc += imm
        return pc + 4
    
    def __jalr(self, rs, imm):
        pc = self.pc
        self.pc = rs + imm
        return pc + 4
    
    
    def __beq(self, rs1, rs2, imm):
        if rs1 == rs2:
            self.pc += imm
        else:
            self.pc += 4
    
    def __bne(self, rs1, rs2, imm):
        if rs1 != rs2:
            self.pc += imm
        else:
            self.pc += 4
    
    def __blt(self, rs1, rs2, imm):
        if rs1 < rs2:
            self.pc += imm
        else:
            self.pc += 4
    
    def __bge(self, rs1, rs2, imm):
        if rs1 >= rs2:
            self.pc += imm
        else:
            self.pc += 4
    
    
    def __lw(self, rs1, imm):
        # print((rs1-0x10000000 + imm)//4)
        return self.data[(rs1-0x10000000 + imm)//4]
    
    def __sw(self, rs1, rs2, imm):
        if self.data == None:
            print(chr(rs2), end="")
        else:
            # print((rs1-0x10000000 + imm)//4)
            self.data[(rs1-0x10000000 + imm)//4] = rs2
    


    def decode_instruction(self, inst):
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
                    imm_se = self.sign_extend(inst >> 20, 12)
                    return f"{instruction} {REGISTERS[rd]}, {imm_se}({REGISTERS[rs1]})"
                case 0x13:
                    if func_3 in {0x1, 0x5}:
                        return f"{instruction} {REGISTERS[rd]}, {REGISTERS[rs1]}, {shamt}"
                    else:
                        imm_se = self.sign_extend(inst >> 20, 12)
                        return f"{instruction} {REGISTERS[rd]}, {REGISTERS[rs1]}, {imm_se}"
                case 0x23:
                    imm_se = self.sign_extend(((inst >> 7) & 0x1F) | ((inst >> 25) << 5), 12)
                    return f"{instruction} {REGISTERS[rs2]}, {imm_se}({REGISTERS[rs1]})"
                case 0x63:
                    imm_se = self.sign_extend(((inst >> 8) & 0xF) << 1 | ((inst >> 25) & 0x3F) << 5 | ((inst >> 7) & 0x1) << 11 | ((inst >> 31) & 0x1) << 12, 13)
                    return f"{instruction} {REGISTERS[rs1]}, {REGISTERS[rs2]}, {imm_se}"
                case 0x37:
                    imm_se = self.sign_extend(inst & 0xFFFFF000, 32)
                    return f"{instruction} {REGISTERS[rd]}, {imm_se}"
                case 0x17:
                    imm_se = inst & 0xFFFFF000
                    return f"{instruction} {REGISTERS[rd]}, {imm_se}"
                case 0x6F:
                    imm_se = self.sign_extend(((inst >> 31) & 0x1) << 20 | ((inst >> 12) & 0xFF) << 12 | ((inst >> 20) & 0x1) << 11 | ((inst >> 21) & 0x3FF) << 1, 21)
                    return f"{instruction} {REGISTERS[rd]}, {imm_se}"
                case 0x67:
                    imm_se = self.sign_extend(inst >> 20, 12)
                    return f"{instruction} {REGISTERS[rd]}, {imm_se}({REGISTERS[rs1]})"
                case _:
                    return f"unknown instruction"
        else:
            return f"unknown instruction"
        
        
    def run_instruction(self, inst):
        opcode = inst & 0x7F
        func_3 = (inst >> 12) & 0x7
        func_7 = (inst >> 25) & 0x7F
        rd = (inst >> 7) & 0x1F
        rs1 = (inst >> 15) & 0x1F
        rs2 = (inst >> 20) & 0x1F
        shamt = (inst >> 20) & 0x1F
        
        if opcode in OPCODES:
            if isinstance(OPCODES[opcode], dict):
                if func_3 in OPCODES[opcode]:
                    if isinstance(OPCODES[opcode][func_3], dict):
                        if func_7 in OPCODES[opcode][func_3]:
                            instruction = OPCODES[opcode][func_3][func_7]
                        else:
                            return False
                    else:
                        instruction = OPCODES[opcode][func_3]
                else:
                    return False
                
            # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7: {hex(func_7)}\trd: {rd}\trs1: {rs1}\trs2: {rs2}\tshamt: {shamt}")
            
            if opcode != 0x23 and rd == 0:
                self.pc += 4
                return True
            match opcode:
                case 0x33:
                    if func_3 == 0x0:
                        if func_7 == 0x00:
                            self.registers[rd] = self.__add(self.registers[rs1], self.registers[rs2])
                        elif func_7 == 0x20:
                            self.registers[rd] = self.__sub(self.registers[rs1], self.registers[rs2])
                    elif func_3 == 0x1:
                        self.registers[rd] = self.__sll(self.registers[rs1], self.registers[rs2])
                    elif func_3 == 0x2:
                        self.registers[rd] = self.__slt(self.registers[rs1], self.registers[rs2])
                    elif func_3 == 0x4:
                        self.registers[rd] = self.__xor(self.registers[rs1], self.registers[rs2])
                    elif func_3 == 0x5:
                        if func_7 == 0x00:
                            self.registers[rd] = self.__srl(self.registers[rs1], self.registers[rs2])
                        elif func_7 == 0x20:
                            self.registers[rd] = self.__sra(self.registers[rs1], self.registers[rs2])
                        else:
                            return False
                    elif func_3 == 0x6:
                        self.registers[rd] = self.__or(self.registers[rs1], self.registers[rs2])
                    elif func_3 == 0x7:
                        self.registers[rd] = self.__and(self.registers[rs1], self.registers[rs2])
                    else:
                        return False
                    self.pc += 4
                    
                case 0x13:
                    if func_3 == 0x0:
                        self.registers[rd] = self.__addi(self.registers[rs1], self.sign_extend(inst >> 20, 12))
                    elif func_3 == 0x1:
                        self.registers[rd] = self.__slli(self.registers[rs1], shamt)
                    elif func_3 == 0x2:
                        self.registers[rd] = self.__slti(self.registers[rs1], self.sign_extend(inst >> 20, 12))
                    elif func_3 == 0x4:
                        self.registers[rd] = self.__xori(self.registers[rs1], self.sign_extend(inst >> 20, 12))
                    elif func_3 == 0x5:
                        if func_7 == 0x00:
                            self.registers[rd] = self.__srli(self.registers[rs1], shamt)
                        elif func_7 == 0x20:
                            self.registers[rd] = self.__srai(self.registers[rs1], shamt)
                        else:
                            return False
                    elif func_3 == 0x6:
                        self.registers[rd] = self.__ori(self.registers[rs1], self.sign_extend(inst >> 20, 12))
                    elif func_3 == 0x7:
                        self.registers[rd] = self.__andi(self.registers[rs1], self.sign_extend(inst >> 20, 12))
                    else:
                        return False
                    self.pc += 4
                    
                case 0x03:
                    # print("check")
                    # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7: {hex(func_7)}\trd: {rd}\trs1: {rs1}\trs2: {rs2}\tshamt: {shamt}")
                    imm_se = self.sign_extend(inst >> 20, 12)
                    if func_3 == 0x2:
                        self.registers[rd] = self.__lw(self.registers[rs1], imm_se)
                    else:
                        return False
                    self.pc += 4
                    
                case 0x23:
                    # print("check")
                    # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7: {hex(func_7)}\trd: {rd}\trs1: {rs1}\trs2: {rs2}\tshamt: {shamt}")
                    imm_se = self.sign_extend(((inst >> 7) & 0x1F) | ((inst >> 25) << 5), 12)
                    if func_3 == 0x2:
                        # print(self.registers[rs1], self.registers[rs2], imm_se)
                        self.__sw(self.registers[rs1], self.registers[rs2], imm_se)
                    else:
                        return False
                    self.pc += 4
                    
                case 0x63:
                    imm_se = self.sign_extend(((inst >> 8) & 0xF) << 1 | ((inst >> 25) & 0x3F) << 5 | ((inst >> 7) & 0x1) << 11 | ((inst >> 31) & 0x1) << 12, 13)
                    if func_3 == 0x0:
                        self.__beq(self.registers[rs1], self.registers[rs2], imm_se)
                    elif func_3 == 0x1:
                        self.__bne(self.registers[rs1], self.registers[rs2], imm_se)
                    elif func_3 == 0x4:
                        self.__blt(self.registers[rs1], self.registers[rs2], imm_se)
                    elif func_3 == 0x5:
                        self.__bge(self.registers[rs1], self.registers[rs2], imm_se)
                    else:
                        return False
                    
                case 0x37:
                    imm20 = (inst >> 12) & 0xFFFFF
                    self.registers[rd] = self.__lui(imm20)
                    self.pc += 4
                    
                case 0x17:
                    self.registers[rd] = self.__auipc(self.pc, inst & 0xFFFFF000)
                    self.pc += 4
                    
                case 0x6F:
                    # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7: {hex(func_7)}\trd: {rd}\trs1: {rs1}\trs2: {rs2}\tshamt: {shamt}")
                    imm = self.sign_extend(((inst >> 31) << 20) | (((inst >> 12) & 0xFF) << 12) | (((inst >> 20) & 0x1) << 11) | (((inst >> 21) & 0x3FF) << 1), 21)
                    self.registers[rd] = self.__jal(imm)
                    
                case 0x67:
                    # print(f"opcode: {hex(opcode)}\tfunc_3: {hex(func_3)}\tfunc_7: {hex(func_7)}\trd: {rd}\trs1: {rs1}\trs2: {rs2}\tshamt: {shamt}")
                    imm = self.sign_extend(inst >> 20, 12)
                    self.registers[rd] = self.__jalr(self.registers[rs1], imm)
                    
                case _:
                    return False
        else:
            return False
        
        return True
    


    def read_binary_file(self, filename):
        with open(filename, "rb") as f:
            binary_data = f.read()
            file_iter = []
            for i in range(0, len(binary_data), 4):
                inst = struct.unpack("<I", binary_data[i:i+4])[0]
                file_iter.append(inst)
            return file_iter
        
    def read_data_file(self, filename):
        with(open(filename, "rb")) as f:
            data = f.read()
            datas = []
            for i in range(0, len(data), 4):
                _data = struct.unpack("<I", data[i:i+4])[0]
                datas.append(_data)
                # print(f"{_data:08x}")
            for _ in range(0xFFFF//4 - len(datas)):
                datas.append(0)
            return datas
        
    
    def write_data_file(self, filename, data):
        with(open(filename, "wb")) as f:
            f.write(data)
        
    
    def print_registers(self):
        for i, _reg in enumerate(self.registers):
            reg = _reg
            if _reg < 0:
                reg = (1 << 32) + _reg
            print(f"{REGISTERS[i]}: 0x{reg:08x}")


    def run(self, _N):
        file_iter = self.read_binary_file(self.filename)
        if self.data != None:
            self.data = self.read_data_file(self.data)
        check = True
        insts = []
        N = int(_N)
        self.pc = 0
        i = 0
        for _, inst in enumerate(file_iter):
            insts.append(inst)
        while self.pc < len(insts)*4 and check and i<N:
            # print(self.pc//4)
            disassembled_inst = self.decode_instruction(insts[self.pc//4])
            check = self.run_instruction(insts[self.pc//4])
            i += 1
            # print(self.pc//4)
            # print(f"inst {i}: {inst:08x} {disassembled_inst}")
            # self.print_registers()
            # print("\n\n")
        self.print_registers()
        if self.data != None:
            # self.write_data_file(self.data, self.data)
            pass
            
            
    def print_binary_file(self, filename):
        file_iter = self.read_binary_file(filename)
        for i, inst in enumerate(file_iter):
            disassembled_inst = self.decode_instruction(inst)
            print(f"inst {i}: {inst:08x} {disassembled_inst}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sim = RiscVSim(sys.argv[1], None)
        if sys.argv[2] == "disassemble":
            sim.print_binary_file(sys.argv[1])
        else:
            sim.run(sys.argv[2])
    elif len(sys.argv) == 4:
        sim = RiscVSim(sys.argv[1], sys.argv[2])
        if sys.argv[2] == "disassemble":
            sim.print_binary_file(sys.argv[1])
        else:
            sim.run(sys.argv[3])
    else:
        print("[!] Error: Usage: python3 riscv-sim.py <binary_file>")
