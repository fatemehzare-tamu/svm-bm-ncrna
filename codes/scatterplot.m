close all
load('cod_rna.mat')
for i=3:9
     figure('WindowState', 'maximized')
index=find(cod_rna(:,1)==-1);

x = cod_rna(index,2);
y = cod_rna(index,i);
sz = 25;
scatter(x,y,sz,[0.8500 0.3250 0.0980])
hold on
index=find(cod_rna(:,1)==1);

x = cod_rna(index,2);
y = cod_rna(index,i);
sz = 25;
hold on
scatter(x,y,sz,[0 0.4470 0.7410])
legend('class : -1', 'class : 1')

txt = ['feature ',num2str(i-1)];
xlabel('feature 1')
ylabel(txt)
% print(string(i),'-dpng')

end

%%
x=[-1,1];
y=[length(find(cod_rna(:,1)==-1)),length(find(cod_rna(:,1)==1))];
bar(x,y)
xlabel('class')

title('Cod-rnd')