import argparse

from accelerate.utils import release_memory, wait_for_everyone


def parse_args():
    parser = argparse.ArgumentParser(description='Task')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--runners', type=str, default='', help='runners of executing task')
    args = parser.parse_args()
    return args


def run_tasks(config, runners):
    from dreamer_train import Tester, Trainer, load_config, utils

    config = load_config(config)
    runners = runners.split(',')
    for runner in runners:
        runner = utils.import_function(runner)
        runner = runner.load(config)
        runner.print(config)
        if isinstance(runner, Trainer):
            runner.save_config(config)
            if config.train.get('resume', False):
                runner.resume()
            runner.train()
        elif isinstance(runner, Tester):
            runner.test()
        else:
            assert False
        release_memory()
        wait_for_everyone()


def main():
    args = parse_args()
    run_tasks(args.config, args.runners)


if __name__ == '__main__':
    main()
