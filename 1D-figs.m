% triangular wave denoising
clear
close all
addpath('/Users/robertnowak/Desktop/Courses/ECE830_Spring15/rwt')

% generate wave function
scale=16;
n=1024/scale;

f=zeros(n,1);


kl=128/scale; % long period
for i=1:n
    p = floor(i/kl);
    fprime(i)=(-1)^p;
    if i>1, 
        f(i) = fprime(i)+f(i-1);
    else
        f(i) = fprime(i)+1;
    end
end
x=(0:n-1)'/n;
plot(x,f)
fl=f;

k=16/scale; % short period
for i=1:n
    p = floor(i/k);
    fprime(i)=(-1)^p;
    if i>1, 
        f(i) = fprime(i)+f(i-1);
    else
        f(i) = fprime(i)+1;
    end
end
x=(0:n-1)'/n;
fs=f;
plot(x,f)

g=fl; a=3;
g(31:47)= kl/k*fs(31:47);
g = g/max(g);
plot(x,g)

% generate data
sigma=.15;
y = g+sigma*randn(size(g));
hold on 
plot(x,y,'r*')
%load triangle_data.mat

lambda = 1.25*sigma;
N=1024;
%wavelet denoise
% h = daubcqf(6);
% J = round(log2(length(y)));
% w = mdwt(y,h);
% 
% nw=zeros(N,1);
% nw(1:length(w)) = sign(w).*max(abs(w)-lambda,0);
% gwave = sqrt(N/n)*midwt(nw,h);
% %gwave = midwt(nw(1:length(w)),h);
% gwave=wshift('1D',gwave,scale);
xx=(0:N-1)/N;
gwave = xx;

% lasso LAR spline
% design matrix
for i=1:length(y)
    X(:,i)= max(x-x(i),0);
end

tol = 10e-5;
lambda = sigma;
alpha = 1/norm(X*X');
delta = 1e6;
btmp = ones(length(X),1);
btmp = pinv(X)*y;
bold = btmp;
%L=0;
% while delta > tol,
%     bold = btmp;
%     tmp = btmp + alpha*(X'*(y-X*btmp));
%     btmp = sign(tmp).*max(abs(tmp)-alpha*lambda,0);
%     delta = norm(btmp-bold)
%     L = L+1
% end
for m=1:2e6,
    bold = btmp;
    tmp = btmp + alpha*(X'*(y-X*btmp));
    btmp = sign(tmp).*max(abs(tmp)-alpha*lambda,0);
    delta = norm(btmp-bold);
    %L = L+1
end
blasso = btmp;
glasso = X*blasso;
% ii=find(abs(blasso)>eps);
% XX=X(:,ii);
% blasso(ii)=pinv(XX)*y;
% blasso = sign(blasso).*max(abs(blasso)-30*lambda,0);
% plot(x,glasso,'k')
gnet = zeros(size(xx));
for i=1:length(glasso)
    gnet = gnet+blasso(i)*max(xx-x(i),0);
end


%  smoothing spline
N=1024; 
theta = fft(y);
L=length(theta);
x=(0:L-1)'/L;
lambda=1e-7;
for k=2:L/2
    damp = 1+lambda*(k-1)^4;
    theta(k) = theta(k)/damp;
    theta(L-k+2) = theta(L-k+2)/damp;
end
theta(L/2+1)=theta(L/2+1)/damp;
fhat = ifft(theta);
plot(fhat)

figure
subplot(411)
plot(x,g)
hold on
plot(x,y,'r.')
title('true function and data')
subplot(412)
plot(x,fhat)
%gss = fit(x,y,'smoothingspline');
%plot((0:N-1)'/N,gss((0:N-1)'/N))
hold on
plot(x,y,'r.')
title('smoothing spline')
subplot(413)
plot(xx,gwave)
hold on
plot(x,y,'r.')
title('wavelet denoising')
subplot(414)
plot(xx,gnet)
hold on
plot(x,y,'r.')
title('neural network')

close all
figure(1)
plot(x,g)
hold on
plot(x,y,'r.','MarkerSize', 10)
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_true.pdf

figure(2)
%plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_sspline_over.pdf

figure(3)
plot(xx,gwave)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_wave_denoise_1.4sigma.pdf

figure(4)
plot(xx,gnet)
hold on
plot(x,y,'r.','MarkerSize', 10)
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_nn_.pdf


%cubic smoothing spline
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.99999999);
%gss = fit(x,y,'smoothingspline');
%[gss,gof,out] = fit(x,y,'cubicinterp');


figure(5)
plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
axis square
print -dpdf fig_sspline_under.pdf

close all

figure(1)
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.99999999);
subplot(211)
plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.','MarkerSize', 10)
set(gca,'xtick',[])
set(gca,'ytick',[])
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.999999);
subplot(212)
plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.','MarkerSize', 10)
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_ssplines.pdf



figure(2)
subplot(211)
plot(xx,gwave)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(212)
plot(xx,gnet)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
print -dpdf fig_wavelet_NN.pdf


save triangle_data_final.mat



% lasso LAR spline
% design matrix
for i=1:length(y)
    X(:,i)= max(x-x(i),0);
end

tol = 10e-5;
lambda = sigma/50;
alpha = 1/norm(X*X');
delta = 1e6;
btmp = ones(length(X),1);
btmp = pinv(X)*y;
bold = btmp;
%L=0;
% while delta > tol,
%     bold = btmp;
%     tmp = btmp + alpha*(X'*(y-X*btmp));
%     btmp = sign(tmp).*max(abs(tmp)-alpha*lambda,0);
%     delta = norm(btmp-bold)
%     L = L+1
% end
for m=1:2e6,
    bold = btmp;
    tmp = btmp + alpha*(X'*(y-X*btmp));
    btmp = sign(tmp).*max(abs(tmp)-alpha*lambda,0);
    delta = norm(btmp-bold);
    %L = L+1
end
blasso = btmp;
glasso = X*blasso;
% ii=find(abs(blasso)>eps);
% XX=X(:,ii);
% blasso(ii)=pinv(XX)*y;
% blasso = sign(blasso).*max(abs(blasso)-30*lambda,0);
% plot(x,glasso,'k')
gnet = zeros(size(xx));
for i=1:length(glasso)
    gnet = gnet+blasso(i)*max(xx-x(i),0);
end

close all
figure(1)
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.99999999);
subplot(411)
plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.999999);
subplot(412)
plot((0:N-1)'/N,gss((0:N-1)'/N))
%plot(x,fhat)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])

subplot(413)
plot(xx,gwave)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(414)
plot(x,g)
hold on
plot(x,y,'r.')
set(gca,'xtick',[])
set(gca,'ytick',[])
%plot(xx,gnet)
%hold on
%plot(x,y,'r.')
%set(gca,'xtick',[])
%set(gca,'ytick',[])
print -dpdf fig_true.pdf



close all
figure(1)
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.99999999);
subplot(311)
plot((0:N-1)'/N,gss((0:N-1)'/N),'linewidth',2)
%plot(x,fhat)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.999999);
subplot(312)
plot((0:N-1)'/N,gss((0:N-1)'/N),'linewidth',2)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(313)
plot(xx,gnet,'linewidth',2)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])

print -djpeg smspline_v_net.jpg
print -dpdf smspline_v_net.pdf




close all
figure(1)
subplot(411)
plot(x,g,'linewidth',2)
hold on
plot(x,y,'r.','MarkerSize', 10)
set(gca,'xtick',[])
set(gca,'ytick',[])
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.99999999);
subplot(412)
plot((0:N-1)'/N,gss((0:N-1)'/N),'linewidth',2)
%plot(x,fhat)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])
[gss,gof,out] = fit(x,y,'smoothingspline','SmoothingParam',0.999999);
subplot(413)
plot((0:N-1)'/N,gss((0:N-1)'/N),'linewidth',2)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(414)
plot(xx,gnet,'linewidth',2)
hold on
plot(x,y,'r.','MarkerSize',10)
set(gca,'xtick',[])
set(gca,'ytick',[])

print -djpeg smspline_v_net.jpg
print -dpdf smspline_v_net.pdf

% compute MSEs

% high sample function
gg = interp(g,16);
plot(g)
