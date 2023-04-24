from ..specNamelist import SPECNamelist
from typing import List, Tuple


def renewInput(inputList: List[str], fileName: str="renewName.txt", reverse: bool=False):
    successIndex, unSuccessIndex = checkCase(inputList)
    renewlist = list()
    for caseIndex in unSuccessIndex:
        for i in range(1, len(inputList)):
            if (caseIndex+i) in successIndex or (caseIndex-i) in successIndex:
                input_namelist = SPECNamelist(inputList[caseIndex]+".end")
                if not reverse:
                    if (caseIndex+i) in successIndex:
                        tem1_namelist = SPECNamelist(inputList[caseIndex+i]+".end")
                        input_namelist["physicslist"]["rac"] = tem1_namelist["physicslist"]["rac"]
                        input_namelist["physicslist"]["zas"] = tem1_namelist["physicslist"]["zas"]
                        input_namelist["physicslist"]["ras"] = tem1_namelist["physicslist"]["ras"]
                        input_namelist["physicslist"]["zac"] = tem1_namelist["physicslist"]["zac"]
                        input_namelist.interface_guess = tem1_namelist.interface_guess
                    else:
                        tem1_namelist = SPECNamelist(inputList[caseIndex-i]+".end")
                        input_namelist["physicslist"]["rac"] = tem1_namelist["physicslist"]["rac"]
                        input_namelist["physicslist"]["zas"] = tem1_namelist["physicslist"]["zas"]
                        input_namelist["physicslist"]["ras"] = tem1_namelist["physicslist"]["ras"]
                        input_namelist["physicslist"]["zac"] = tem1_namelist["physicslist"]["zac"]
                        input_namelist.interface_guess = tem1_namelist.interface_guess
                else:
                    if (caseIndex-i) in successIndex:
                        tem1_namelist = SPECNamelist(inputList[caseIndex+i]+".end")
                        input_namelist["physicslist"]["rac"] = tem1_namelist["physicslist"]["rac"]
                        input_namelist["physicslist"]["zas"] = tem1_namelist["physicslist"]["zas"]
                        input_namelist["physicslist"]["ras"] = tem1_namelist["physicslist"]["ras"]
                        input_namelist["physicslist"]["zac"] = tem1_namelist["physicslist"]["zac"]
                        input_namelist.interface_guess = tem1_namelist.interface_guess
                    else:
                        tem1_namelist = SPECNamelist(inputList[caseIndex-i]+".end")
                        input_namelist["physicslist"]["rac"] = tem1_namelist["physicslist"]["rac"]
                        input_namelist["physicslist"]["zas"] = tem1_namelist["physicslist"]["zas"]
                        input_namelist["physicslist"]["ras"] = tem1_namelist["physicslist"]["ras"]
                        input_namelist["physicslist"]["zac"] = tem1_namelist["physicslist"]["zac"]
                        input_namelist.interface_guess = tem1_namelist.interface_guess
                input_namelist["screenlist"]["Wpp00aa"] = True
                input_namelist.write(filename=inputList[caseIndex], force= True)
                renewlist.append(caseIndex)
                print("Rewrite the input file " + inputList[caseIndex] + "! ")
                break
    print(renewlist)
    if fileName != None:
        with open(fileName, 'w') as f:
            for index in renewlist:
                f.write(inputList[index] + "\n")



def checkCase(inputList: List[str]) -> Tuple[List]:
    successIndex = list()
    unSuccessIndex = list()
    for index, file in enumerate(inputList):
        state = False
        file = file.replace(".sp", ".log")
        with open(file, 'r') as f:
            for line in f:
                if "finished" in line and "success" in line:
                    successIndex.append(index)
                    state = True
                    break
            if not state:
                unSuccessIndex.append(index)
    return successIndex, unSuccessIndex


if __name__ == "__main__":
    pass
