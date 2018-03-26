import CustomBVHWhitelistCreation as wCreator
import ModelEvaluationTools as met
import MKZIntentConf as conf
import os

if __name__ == '__main__':
    logger = met.create_logger("driver_log", conf.LOG_DIR)

    # create whitelist
    if not os.path.exists(conf.WHITE_LIST_FILE):
        logger.info("Whitelist file not found, creating whitelist...")
        wCreator.main()
        logger.info("Whitelist generated as " + os.getcwd() + conf.WHITE_LIST_FILE)
    else:
        logger.info("Found whitelist file as " + conf.WHITE_LIST_FILE)