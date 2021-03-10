import logging


class Logger:
    def __init__(self, title="Logger:", filename="Default.log"):
        # 1.生成记录器
        self.logger = logging.getLogger(__name__)
        # 2.记录器配置
        logging.basicConfig(format="%(asctime)s - %(filename)s - %(lineno)s - %(message)s",
                            level=logging.DEBUG,
                            filename=filename,
                            filemode='w')
        # 3.输出到控制台
        console = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(lineno)s - %(message)s")
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # 记录标题
        self.logger.info(title)

    def write_info(self, message):
        # 记录info
        self.logger.info(message)

    def write_warning(self, message):
        # 记录warning
        self.logger.warning(message)


logger = Logger(title="This is logger test:", filename="logger.log")
for i in range(10):
    logger.write_info("%d" % i)






