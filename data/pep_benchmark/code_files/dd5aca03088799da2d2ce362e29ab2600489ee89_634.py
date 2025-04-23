# -*- coding: utf-8 -*-
import os
import platform
import sys
import shutil

buildDir = "build/src/"
binDir = "bin/"
targets = ("HelloWorld", "CreateWindow", "Triangle",
           "Shader", "Texture", "Transformation", "Coordinate")
targetDeps = {}
system = platform.system()


def clean():
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("bin"):
        shutil.rmtree("bin")


def build(buildType):
    clean()
    os.mkdir("build")
    buildFlag = True

    if system == "Windows":
        buildFlag = os.system(
            'cd build && cmake -G "Visual Studio 16 2019" .. && cmake --build . --config {}'.format(buildType))
    else:
        buildFlag = os.system(
            'cd build && cmake .. -DCMAKE_BUILD_TYPE={} && cmake --build . -j8'.format(buildType))

    if buildFlag:
        print("build failed")
        return False

    os.mkdir("bin")

    def getTargetPath(s):
        if system == "Windows":
            return (buildDir + s + "/{}/" + s + ".exe").format(buildType)
        return buildDir + s + "/" + s

    for i in targets:
        shutil.move(getTargetPath(i), binDir)

    shutil.rmtree("build")
    shutil.copytree("res", binDir + "res")

    print("build done")
    return True


def printHelp():
    print("usage: [build] [run <target>] [clean] [--help]")
    print("available targets are:")
    print(targets)


def runTarget(target):
    if system == "Windows":
        os.system("cd " + binDir + " && " + target)
    else:
        os.system("cd " + binDir + " && ./" + target)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        printHelp()
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "--help" or cmd == "-h":
        printHelp()
    elif cmd == "build":
        status = True
        if len(sys.argv) == 3 and sys.argv[2] == "Debug":
            status &= build("Debug")
        else:
            status &= build("Release")
        if not status:
            sys.exit(-1)
    elif cmd == "run":
        if len(sys.argv) < 3:
            print("please input a target to run, available targets are:")
            print(targets)
            sys.exit(0)
        runTarget(sys.argv[2])
    elif cmd == "clean":
        clean()
    else:
        print("unknown option " + cmd)
        printHelp()