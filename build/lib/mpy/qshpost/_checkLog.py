from typing import List


def checkLog(specFile: List[str] or List[List[str]], logFile: str="log.txt", successWrite: str=None) -> List[str]:

    if isinstance(specFile[0], str):
        log = list()
        successList = list()
        for file in specFile:
            file = file.replace("sp", "log")
            with open(file, 'r') as f:
                log.append("=============================================================\n")
                log.append(file+": \n")
                for line in f:
                    if "finished" in line:
                        if "success" in line:
                            successList.append(file.replace(".log",".sp"))
                        log.append(line)
                        log.append(f.readline())
                        log.append(f.readline())
                        log.append("\n")
                        continue
        with open(logFile, "w") as f:
            for line in log:
                f.write(line)
    elif isinstance(specFile[0], List) and isinstance(specFile[0][0], str):
        log = list()
        successList = list()
        for fileList in specFile:
            _sList = list()
            for file in fileList:
                file = file.replace("sp", "log")
                with open(file, 'r') as f:
                    log.append("=============================================================\n")
                    log.append(file+": \n")
                    for line in f:
                        if "finished" in line:
                            if "success" in line:
                                _sList.append(file.replace(".log",".sp"))
                            log.append(line)
                            log.append(f.readline())
                            log.append(f.readline())
                            log.append("\n")
                            continue
            successList.append(_sList)
        with open(logFile, "w") as f:
            for line in log:
                f.write(line)
    else:
        raise TypeError (
            "Check the type of specFile! "
        )

    if successWrite != None:
        with open(successWrite, 'w') as f:
            for line in successList:
                if isinstance(line, str):
                    f.write(line + "\n")
                if isinstance(line, List):
                    for file_name in line:
                        f.write(file_name + "\n")

    return successList

    