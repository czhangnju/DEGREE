function [G] = DEGREE(XT, param)
%   min  alpha*\sum_{i~=j,i<j} |Vi'Vj|^2 + |H'Vj|^2 + beta*|C|_*   + gamma*|D|_{t*}
%         + lamba1*|E1|_{2,1} + lambda2*|E2|_{2,1} + lambda3*|E3|_{2,1}        
%   s.t.  Xi = Pi(H+Vi) + E1^i, H = HS + E2, S=C,Vi=ViQi+E3^i, Qi=Di, PiPi^T = I

num_view = length(XT);
X = []; 
X = []; dk=[]; Xc=[];
for i=1:num_view
    dk{i} = size(XT{i},1);
    X{i} = XT{i}./repmat(sqrt(sum(XT{i}.^2,1)),size(XT{i},1),1);  %normalized
    Xc = [Xc; X{i}];
end
[~, n] = size(X{1});

alpha  =  param.alpha;
gamma =  param.gamma;
beta    = param.beta;
lambda  =  param.lambda;
dim  = param.dim;

MAX_iter = 100; 

options = [];
options.ReducedDim = dim;
[W,~] = PCA1(Xc',options);     
H = W'*Xc;


E1 = cell(1,num_view); V = E1; 
for iv = 1:num_view
    E1{iv} = zeros(dk{iv}, n);
    V{iv}= zeros(dim, n);
    Z5{iv} = zeros(n);
    Q{iv} = zeros(n);
end
S =  zeros(n); 
D = Q;
E2 = zeros(dim,n);
E3 = V;

Z1 = E1; 
Z2 = zeros(dim, n);
Z3 = zeros(n);
Z4 = E3;

rho = 1e-3;
theta = 1.1;

for  iter = 1:MAX_iter
    % P step
    for i=1:num_view
        Temp = X{i} - E1{i} + Z1{i}/rho;
        [U1,~,V1] = svd((H+V{i})*Temp','econ');
        P{i} = V1*U1';
       clear Temp U1 V1;
    end
    
    % H step
    HA = 0;  HC = 0;
    for i=1:num_view
        HA = HA + rho*P{i}'*P{i} + 2*alpha*V{i}*V{i}';
        HC = HC + rho*P{i}'*(X{i}-P{i}*V{i}-E1{i}+Z1{i}/rho);
    end
    HC = HC + rho*(E2-Z2/rho)*(eye(n)-S)';
    HB = rho*(eye(n)-S)*(eye(n)-S)';
    H = lyap(HA,HB,-HC);
    clear HA HB HC;
         
     % V step
     for i=1:num_view
         KK=0;
         for j=1:num_view
             if j == i
                 continue;
             end  
             KK = KK + V{j}*V{j}';
         end
         tVA = 2*alpha*(H*H'+KK) + rho*P{i}'*P{i};
         tVB = rho*(eye(n)-Q{i})*(eye(n)-Q{i})';
         tVC = rho*P{i}'*(X{i}-P{i}*H-E1{i}+Z1{i}/rho)+rho*(E3{i}-Z4{i}/rho)*(eye(n)-Q{i})';
          V{i} = lyap(tVA,tVB,-tVC);
     end
      clear tVA tVB tVC KK
      
    % E step
    for i=1:num_view
        E1{i} = prox_l21(X{i}-P{i}*(H+V{i}) + Z1{i}/rho,1/rho);
        E3{i} = prox_l21(V{i}-V{i}*Q{i} + Z4{i}/rho,lambda/rho);
    end
     E2 = prox_l21(H - H*S + Z2/rho,lambda/rho);
    
    
     % C step
    C = prox_nuclear(S + Z3/rho, beta/rho); 
    
    %D
    Q_tensor = cat(3, Q{:,:});
    Z5_tensor = cat(3, Z5{:,:});
    Qv = Q_tensor(:);
    Z5v = Z5_tensor(:);
    [Dv, ~] = wshrinkObj(Qv + 1/rho*Z5v, gamma/rho, [n, n, num_view], 0,3);
    D_tensor = reshape(Dv, [n, n, num_view]);  
    for i=1:num_view
        D{i} = D_tensor(:,:,i);
    end
      
     % Q step
    for i=1:num_view
    tQ1 = rho*V{i}'*V{i} + rho*eye(n);
    tQ2 = rho*V{i}'*(V{i}-E3{i}+Z4{i}/rho)+ rho*D{i} - Z5{i};
    Q{i} = tQ1\tQ2;
    end
    clear tQ1 tQ2
 
            % S step
    tS1 = rho*H'*H + rho*eye(n);
    tS2 = rho*H'*(H-E2+Z2/rho)+rho*C-Z3;
    S = tS1\tS2;
    clear tS1 tS2
    
    %
    RR=[];RR2=[]; RR3=[];
    for i = 1:num_view
        res = X{i} - P{i}*(H+V{i}) - E1{i};
        res2 = V{i} - V{i}*Q{i} - E3{i};
        res3 = Q{i} - D{i};
        Z1{i} = Z1{i} + rho*res;
        Z4{i} = Z4{i} + rho*res2;
        Z5{i} = Z5{i} + rho*res3;
        RR=[RR; res];
        RR2=[RR2; res2];
        RR3=[RR3; res3];
    end
    Z2 = Z2 + rho*(H-H*S-E2);
    Z3 = Z3 + rho*(S-C);
 
    rho = min(1e6, theta*rho);   
    thrsh = 1e-3;
    if(norm(RR, inf)<thrsh && norm(RR2, inf)<thrsh && norm(RR3, inf)<thrsh && norm(H-H*S-E2,inf)<thrsh && norm(S-C,inf)<thrsh)
        break;
    end
    
end

KK = 0;
for i=1:num_view
    KK = KK + (abs(Q{i})+(abs(Q{i}))')/2;
end
G = (abs(S)+(abs(S))')/2 + KK/num_view;

end

