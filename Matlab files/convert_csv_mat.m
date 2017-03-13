function [] = convert_csv_mat()
    aerosol_full = csvread('aerosol_full.csv');
    %Y = csvread('MODIS_y.csv');
    %nb_bag = 1364;
    %X = cell(nb_bag, 1);
    
    %for i = 1:nb_bag
     %   index_start = 1 + 100 * (i-1);
      %  index_end = 100 * i;
       % X{i, 1} = transpose(M(index_start:index_end, 2:13));
    %end
        
    save('aerosol_full.mat', 'aerosol_full');