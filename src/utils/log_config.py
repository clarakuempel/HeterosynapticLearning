
def prepare_config(full_conf):
    config = {}
    config["model"] = full_conf["model"]
    config["data"] = full_conf["data"]
    config["trainer"] = full_conf["trainer"]
    return config
