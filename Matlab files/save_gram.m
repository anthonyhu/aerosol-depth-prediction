gram_file = 'datasetaerosol_full_Gram_costexpected_kratquadr_log2kp10';
output_name = 'rquadr_1024.csv';

G = load(gram_file);
dlmwrite(output_name, G.G, 'delimiter', ',', 'precision', 15);