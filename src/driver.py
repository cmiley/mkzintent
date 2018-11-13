import CustomBVHWhitelistCreation as wCreator
import ModelEvaluationTools as Met
import MKZIntentConf as Conf
import os

if __name__ == '__main__':
    logger = Met.create_logger("driver_log", Conf.LOG_DIR)

    # create whitelist
    if not os.path.exists(Conf.WHITE_LIST_FILE):
        logger.info("Whitelist file not found, creating whitelist...")
        wCreator.main()
        logger.info("Whitelist generated as " + os.getcwd() + "/" + Conf.WHITE_LIST_FILE)
    else:
        logger.info("Found whitelist file as " + Conf.WHITE_LIST_FILE)
