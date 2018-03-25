import CustomBVHWhitelistCreation as wCreator
import os

if __name__ == '__main__':
    src = os.path.join(os.getcwd(), "src/")

    # create whitelist
    if not os.path.exists("white_list"):
        wCreator.main()