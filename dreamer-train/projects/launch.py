import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import ENV

class DreamerOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="options for dreamer project")
        self.parser.add_argument('--project_name', 
                                help='project_name is the filename under projects, you can customize various projects under this folder',
                                type=str,
                                default='DriveDreamer')
        self.parser.add_argument('--config_name', 
                                help='config_name is the filename under projects/project_name/configs, you can customize various configs (data, train, test, models, optimizer, schedulers, etc) under this folder',
                                type=str,
                                default='drivedreamer-img_sd15_corners_hdmap_res448')
        self.parser.add_argument('--runners', 
                                help='runners is a list, you can specify multiple runners, like [drivedreamer.DriveDreamerTrainer,drivedreamer.DriveDreamerTester]',
                                type=str,
                                nargs='+',
                                default=['drivedreamer.DriveDreamerTrainer'])
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

def main():
    opts = DreamerOptions().parse()
    ENV.init_paths(project_name=opts.project_name)
    from dreamer_train import launch_from_config
    config_path = 'configs.{}.config'.format(opts.config_name)
    launch_from_config(config_path, ','.join(opts.runners))

if __name__ == '__main__':
    main()
