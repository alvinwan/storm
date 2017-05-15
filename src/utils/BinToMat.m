% This script (based on readbinfileNXcYcZc) converts BIN files to MAT files
% for use with Python training scripts

% Replace the following with the BIN file path
inpath = '/home/cc/cs61b/sp17/staff/cs61b-taf/1_zc-CropROIs-average-match.bin';
% Replace the following with the MAT file path
outpath = 'out.mat';

fid = fopen(inpath,'r');
c = fread(fid,12);
A = fread(fid,'*single');


%   1       2   3   4   5   6       7       8       9   10  11  12
% Cas44178      X       Y       Xc      Yc      Height  Area    Width   Phi     Ax      BG      I       
%   13  14      15      16      17  18
% Frame Length  Link    Valid   Z       Zc

x = (A(2:18:end));
y = (A(3:18:end));
xc = (A(4:18:end));
yc = (A(5:18:end));
z = (A(18:18:end));
zc = (A(19:18:end));
h = (A(6:18:end));
area = (A(7:18:end));
width = (A(8:18:end));
phi = (A(9:18:end));
Ax = (A(10:18:end));
bg = (A(11:18:end));
I = (A(12:18:end));

clear A;
% ReadInt

frewind(fid);

c = fread(fid,3,'*int32');
TotalFrames=c(2);

A = fread(fid,'*int32');

N=A(1);
cat = (A(13:18:end));
valid = (A(14:18:end));
frame = (A(15:18:end));
length = (A(16:18:end));
link=(A(17:18:end));
clear A;

fclose(fid);

%N = min([ size(cat,1),size(z,1),size(bg,1),size(area,1) ]);
ind = 1:N;
%N = size(ind,1);
% mol = struct('cat',{cat(ind)},'x',{x(ind)},'y',{y(ind)},'z',{z(ind)},'h',{h(ind)},'area',{area(ind)},'width',{width(ind)},'phi',{phi(ind)},         'Ax',{Ax(ind)},'bg',{bg(ind)},'I',{I(ind)},'frame',{frame(ind)},'length',{length(ind)},'link',{-1*ones(N,1)},'valid',{valid(ind)});
mol.cat = cat(ind);
mol.x = x(ind);
mol.y = y(ind);
mol.z = z(ind);
mol.xc = xc(ind);
mol.yc = yc(ind);
mol.zc = zc(ind);
mol.h = h(ind);
mol.area = area(ind);
mol.width = width(ind);
mol.phi = phi(ind);
mol.Ax = Ax(ind);
mol.bg = bg(ind);
mol.I = I(ind);
mol.frame = frame(ind);
mol.length = length(ind);
mol.link = link(ind);
mol.valid = valid(ind);

mol.N=N;
mol.TotalFrames=TotalFrames;

save(outpath, '-struct', 'mol')
