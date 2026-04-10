from recbole.quick_start import run_recbole
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    run_recbole(
        model=args.model,
        dataset='amazon_m2',
        config_file_list=['scripts/config.yaml']
    )
    
if __name__=="__main__":
    main()