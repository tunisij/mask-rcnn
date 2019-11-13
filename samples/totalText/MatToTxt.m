myFiles = dir(fullfile('C:\Users\tunis\Documents\UofM\ECE5831\FinalProject\TT_new_train_GT\Train','*.mat'));
for k = 1:length(myFiles)
  fileName = myFiles(k).name;
  path = fullfile('C:\Users\tunis\Documents\UofM\ECE5831\FinalProject\TT_new_train_GT\Train\', fileName);
  
  data = load(path);
  list = [];
  
  for j=1:size(data.gt, 1)
      x = data.gt{j,2};
      y = data.gt{j,4};
      list = [list, jsonencode(table(x,y))];
  end

  fid=fopen("C:\Users\tunis\Documents\UofM\ECE5831\FinalProject\TT_new_train_GT\TrainCsv\" + fileName + ".txt",'w');
  fprintf(fid, list);
  fclose(fid);
  
end