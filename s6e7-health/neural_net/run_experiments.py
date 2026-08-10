import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "neural_net"))
from train_nn import train_nn

def main():
    experiments = [
        {"loss": "weighted_ce", "epochs": 15, "batch_size": 2048, "lr": 0.003},
        {"loss": "weighted_focal", "epochs": 15, "batch_size": 2048, "lr": 0.003, "gamma": 2.0},
        {"loss": "focal", "epochs": 15, "batch_size": 2048, "lr": 0.003, "gamma": 2.0},
        {"loss": "ce", "epochs": 15, "batch_size": 2048, "lr": 0.003}
    ]
    
    results = []
    
    for i, exp in enumerate(experiments):
        loss_type = exp["loss"]
        epochs = exp["epochs"]
        batch_size = exp["batch_size"]
        lr = exp["lr"]
        gamma = exp.get("gamma", 2.0)
        
        prefix = f"exp_{loss_type}"
        print(f"\n==================================================")
        print(f"RUNNING EXPERIMENT {i+1}/{len(experiments)}: {loss_type}")
        print(f"==================================================")
        
        try:
            ba_argmax, ba_tuned, w, oof, test_preds = train_nn(
                loss_type=loss_type,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                gamma=gamma,
                dry_run=False,
                save_prefix=prefix
            )
            
            results.append({
                "loss": loss_type,
                "ba_argmax": ba_argmax,
                "ba_tuned": ba_tuned,
                "weights": w,
                "prefix": prefix
            })
            
            print(f"Finished Experiment {loss_type}: Argmax={ba_argmax:.4f}, Tuned={ba_tuned:.4f}")
            
        except Exception as e:
            print(f"Experiment {loss_type} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n==================================================")
    print("ALL EXPERIMENTS COMPLETED. SUMMARY:")
    print("==================================================")
    print(f"{'Loss Type':<16} | {'Argmax BA':<10} | {'Tuned BA':<10} | {'Tuned Weights'}")
    print("-" * 70)
    for res in results:
        weights_str = "[" + ", ".join([f"{val:.3f}" for val in res["weights"]]) + "]"
        print(f"{res['loss']:<16} | {res['ba_argmax']:.5f}  | {res['ba_tuned']:.5f}  | {weights_str}")
    
    if len(results) == 0:
        print("No successful experiments.")
        return
        
    # Pick the best experiment based on tuned balanced accuracy
    best_exp = max(results, key=lambda x: x["ba_tuned"])
    print(f"\nBest Experiment: {best_exp['loss']} with Tuned OOF Bal-Acc = {best_exp['ba_tuned']:.5f}")
    
    # Copy best predictions to oof.csv and test_proba.csv
    src_oof = ROOT / "neural_net" / f"{best_exp['prefix']}_oof.csv"
    src_test = ROOT / "neural_net" / f"{best_exp['prefix']}_test_proba.csv"
    dst_oof = ROOT / "neural_net" / "oof.csv"
    dst_test = ROOT / "neural_net" / "test_proba.csv"
    
    shutil.copy(src_oof, dst_oof)
    shutil.copy(src_test, dst_test)
    
    print(f"Copied best predictions:")
    print(f"  OOF:  {src_oof.name} -> {dst_oof.name}")
    print(f"  Test: {src_test.name} -> {dst_test.name}")
    print("\nDeliverables written successfully.")

if __name__ == "__main__":
    main()
