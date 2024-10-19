#!/bin/bash -l
#SBATCH -J SEGNW-test
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=16GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:25:00 
## Nazwa grantu do rozliczenia zużycia zasobów CPU
#SBATCH -A plgsegnw-gpu-a100
## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
## Liczba GPU
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="output_time_right.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error_right.err"

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun python train_model_right.py
