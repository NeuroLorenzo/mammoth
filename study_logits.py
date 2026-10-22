import os
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_logits_across_tasks(save_path, num_tasks=5):
    """
    Legge i file .pkl generati durante l'evaluation e plotta la media dei logit 
    prodotti dalla rete per ciascun task.
    """

    cil_accs = []
    til_accs = []
    for k in range(0, num_tasks):
        task_logit_means = []
        
        cil_acc = []
        til_acc = []
        for i in range(0,k+1):
            filename = f"{save_path}/task_{(k+1)*2}_{i}_eval.pkl"
            print(filename)
            if not os.path.exists(filename):
                print(f"File non trovato: {filename}")
                continue

            all_logits = []
            
            # Gestisce i pickle salvati in modalità 'ab' (append)
            with open(filename, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        all_logits.append(data['logits'])
                        cil_acc.append(data[f'accs'][i])
                        til_acc.append(data[f'accs_mask_classes'][i])
                    except EOFError:
                        break  # Fine del file
            # print(f'logits {k} {i} : ',data['logits'])
            

            if len(all_logits) == 0:
                continue
            if k>=4:

                # Concateniamo tutti gli array di logit accumulati nel file
                # Shape finale: [N_campioni, N_classi_totali]
                concat_logits = np.concatenate(all_logits, axis=0)
                # print(all_logits.shape)
                # print(concat_logits.shape)
                # Calcoliamo la media dei logit lungo la dimensione dei campioni (dim=0)
                # Risultato: un array 1D con il logit medio per ogni classe [mean_c0, mean_c1, ...]
                mean_logits = np.mean(concat_logits, axis=0)
                task_logit_means.append(mean_logits)

                print(f"Task {k} - Logit medi per classe: {np.round(mean_logits, 3)}")
                print(f"Task {k} - accs: {cil_accs}")
                print(f"Task {k} - accs_mask: {til_accs}")

        cil_accs.append(cil_acc)
        til_accs.append(til_acc)
        if k>=4:

            # --- PLOTTING ---
            plt.figure(figsize=(10, 6))
            
            num_classes = len(task_logit_means[0]) if task_logit_means else 0
            epochs_eval = range(1, len(task_logit_means) + 1)

            # Convertiamo in matrice 2D per comodità: [Task_eval, Classi]
            task_logit_means = np.array(task_logit_means)

            for c in range(num_classes):
                # Tracciamo l'evoluzione della classe 'c' al trascorrere dei task
                plt.plot(epochs_eval, task_logit_means[:, c], marker='o', linewidth=2, label=f'Classe {c}')

            plt.title('Evoluzione della Media dei Logit per Classe al progredire dei Task')
            plt.xlabel('Task Valutato / Completato')
            plt.ylabel('Valore Medio del Logit')
            plt.xticks(epochs_eval)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{save_path}/logits_evolution.png", bbox_inches='tight')

            cmap=plt.get_cmap('tab10')
            c=[cmap(i) for i in np.linspace(0,1,5)]
            plt.figure()
            plt.title('Task Incremental Accuracy')
            plt.xlabel('tasks')
            plt.ylabel('accuracy')
            for i in range(5):
                k=[l[i] for l in til_accs[i:]]
                print(i,k)

                plt.plot(np.arange(i+1,5+1),k,'*-.')
            plt.savefig(f"{save_path}/til_accs.png", bbox_inches='tight')
                
            plt.figure()
            plt.title('Class Incremental Accuracy')
            plt.xlabel('tasks')
            plt.ylabel('accuracy')
            for i in range(5):
                k=[l[i] for l in cil_accs[i:]]
                plt.plot(np.arange(i+1,5+1),k,'*-.')
            plt.savefig(f"{save_path}/cil_accs.png", bbox_inches='tight')
    # --- ESECUZIONE ---
# Inserisci il tuo save_path qui
analyze_logits_across_tasks('debug_logits', num_tasks=5)