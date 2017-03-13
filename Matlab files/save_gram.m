gram_file = 'datasetaerosol_full_Gram_costexpected_kCauchy_log2kp4.mat';
output_name = 'Cauchy_16.csv';

G = load(gram_file);
dlmwrite(output_name, G.G, 'delimiter', ',', 'precision', 15);