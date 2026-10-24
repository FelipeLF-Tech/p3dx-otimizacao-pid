function resultado = atividade4_pid_ga_rapida()
% ATIVIDADE4_PID_GA_RAPIDA
% Compara fmincon e GA em Go-to-Goal no CoppeliaSim, com custos iguais.
% Menos simulações por execução: GA com população 8, no máximo 8 gerações,
% parada por estagnação em 3 gerações. Duas sementes são usadas como teste.
% A configuração é um compromisso para execução rápida; não garante ótimo
% global nem convergência. Aumente gerações/população se o tempo permitir.
%
% Requisitos: MATLAB Optimization Toolbox, Global Optimization Toolbox,
% CoppeliaSim aberto com cena Pioneer P3DX e API ZeroMQ para MATLAB.

if nargin > 0, error('A função não recebe argumentos.'); end
cfg.api = 'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\matlab';
cfg.L = 0.381; cfg.R = 0.0975; cfg.v = 0.12; cfg.N = 50;
cfg.x0 = [1 0.1 0.01]; cfg.lb = [0.1 0 0]; cfg.ub = [10 1 1];
cfg.goalTol = 0.1; cfg.maxSteps = 500; cfg.wmax = 2; cfg.iMax = 2;
cfg.pop = 8; cfg.maxGen = 8; cfg.maxStall = 3;
cfg.seeds = [11 22]; cfg.penalty = 1e4;
cfg.outdir = fullfile(fileparts(mfilename('fullpath')), ...
    ['resultados_rapidos_' datestr(now,'yyyymmdd_HHMMSS')]);
mkdir(cfg.outdir);
assert(isfolder(cfg.api),'Ajuste cfg.api para o caminho da ZeroMQ API do CoppeliaSim.');
addpath(cfg.api);
assert(exist('ga','file')==2 && exist('fmincon','file')==2, ...
    'Instale/habilite Optimization Toolbox e Global Optimization Toolbox.');

client = RemoteAPIClient('host','127.0.0.1','port',23000);
sim = client.require('sim');
assert(scalar(sim.getSimulationState())==sim.simulation_stopped, ...
    'Pare a simulação e restaure a cena inicial antes de começar.');
body=sim.getObject('/Pioneer_p3dx');
left=sim.getObject('/Pioneer_p3dx_leftMotor');
right=sim.getObject('/Pioneer_p3dx_rightMotor');
goal=sim.getObject('/Goal');
cfg.p0=vector(sim.getObjectPosition(body,sim.handle_world));
cfg.o0=vector(sim.getObjectOrientation(body,sim.handle_world));
cfg.pg=vector(sim.getObjectPosition(goal,sim.handle_world));
cfg.h=scalar(sim.getSimulationTimeStep());
assert(norm(cfg.pg(1:2)-cfg.p0(1:2))>cfg.goalTol,'Robô já está no alvo.');
cleanup=onCleanup(@() safeStop(sim,left,right)); %#ok<NASGU>

% Primeiro verifica se uma simulação é rápida e o cenário responde.
t0=tic; base=simulate(cfg.x0); tBase=toc(t0);
if ~base.reached
    error('Controlador inicial não alcançou o alvo em %.1f s simulados. Confira cena/horizonte.',cfg.maxSteps*cfg.h);
end
save(fullfile(cfg.outdir,'baseline.mat'),'base','cfg');
writetable(poseTable(base),fullfile(cfg.outdir,'pose_inicial.csv'));
fprintf('Uma avaliação: %.1f s reais. Estimativa grosseira: aprox. %.1f min para duas buscas locais + dois GAs.\n', ...
    tBase, (2*(8*8+40)+2*500)*tBase/60);

rows=cell(0,12); runs=struct([]); run=0;
for criterion=1:2
    % Mesmo modelo/custo para ambos. criterion=1 ITSE; criterion=2 max desvio.
    nEval=0; history=[]; evalLog=[];
    objective=@cost;
    fopts=optimoptions('fmincon','Algorithm','sqp','Display','off', ...
        'MaxIterations',12,'MaxFunctionEvaluations',60, ...
        'OptimalityTolerance',2e-3,'StepTolerance',2e-3,'UseParallel',false, ...
        'OutputFcn',@fout);
    [g,j,flag,out]=fmincon(objective,cfg.x0,[],[],[],[],cfg.lb,cfg.ub,[],fopts);
    run=run+1; runs(run)=record('fmincon',criterion,NaN,g,j,flag,out,history,evalLog); %#ok<AGROW>
    appendRow(runs(run));

    for seed=cfg.seeds
        rng(seed,'twister'); nEval=0; history=[]; evalLog=[];
        gopts=optimoptions('ga','Display','off','PopulationSize',cfg.pop, ...
            'MaxGenerations',cfg.maxGen,'MaxStallGenerations',cfg.maxStall, ...
            'FunctionTolerance',2e-3,'EliteCount',1,'CrossoverFraction',0.75, ...
            'InitialPopulationMatrix',cfg.x0,'UseParallel',false, ...
            'UseVectorized',false,'OutputFcn',@gout);
        [g,j,flag,out]=ga(objective,3,[],[],[],[],cfg.lb,cfg.ub,[],gopts);
        run=run+1; runs(run)=record('GA',criterion,seed,g,j,flag,out,history,evalLog); %#ok<AGROW>
        appendRow(runs(run));
    end
    plotCriterion(criterion);
end
T=cell2table(rows,'VariableNames',{'Metodo','Criterio','Semente','Kp','Ki','Kd', ...
    'CustoBusca','CustoValidado','ITSE','DesvioMax_m','DistFinal_m','Avaliacoes'});
writetable(T,fullfile(cfg.outdir,'resumo.csv'));
save(fullfile(cfg.outdir,'resultados.mat'),'cfg','base','runs','T');
fprintf('Finalizado. Resumo e gráficos em: %s\n',cfg.outdir);

    function J=cost(gains)
        nEval=nEval+1; tick=tic;
        d=simulate(gains);
        if criterion==1, raw=d.ITSE; else, raw=d.Jline; end
        J=raw;
        if ~d.reached, J=cfg.penalty+d.finalDist; end
        evalLog(end+1,:)=[nEval gains(:)' raw J double(d.reached) d.finalDist toc(tick)]; %#ok<AGROW>
    end
    function stop=fout(x,ov,state)
        stop=false;
        if strcmp(state,'init')||strcmp(state,'iter')
            history(end+1,:)=[ov.iteration ov.fval x(:)' nEval]; %#ok<AGROW>
        end
    end
    function [state,options,optchanged]=gout(options,state,flag)
        optchanged=false;
        if strcmp(flag,'init')||strcmp(flag,'iter')
            [best,k]=min(state.Score);
            history(end+1,:)=[state.Generation best state.Population(k,:) nEval]; %#ok<AGROW>
        end
    end
    function r=record(method,crit,seed,g,j,flag,out,hist,ev)
        % Uma simulação final de validação não entra na contagem do otimizador.
        final=simulate(g);
        r=struct('method',method,'criterion',crit,'seed',seed,'gains',g, ...
            'searchCost',j,'exitflag',flag,'output',out,'history',hist, ...
            'evaluations',ev,'nEval',size(ev,1),'final',final);
        tag=sprintf('%s_c%d_s%g',method,crit,seed);
        writetable(poseTable(final),fullfile(cfg.outdir,['pose_' tag '.csv']));
        if ~isempty(hist)
            writetable(array2table(hist,'VariableNames',{'IteracaoGeracao','Custo','Kp','Ki','Kd','Avaliacoes'}), ...
                fullfile(cfg.outdir,['historico_' tag '.csv']));
        end
        if ~isempty(ev)
            writetable(array2table(ev,'VariableNames',{'Avaliacao','Kp','Ki','Kd','CustoBruto','CustoPenalizado','Chegou','DistFinal_m','Tempo_s'}), ...
                fullfile(cfg.outdir,['avaliacoes_' tag '.csv']));
        end
    end
    function appendRow(r)
        if r.criterion==1, validCost=r.final.ITSE; else, validCost=r.final.Jline; end
        rows(end+1,:)={r.method,r.criterion,r.seed,r.gains(1),r.gains(2),r.gains(3), ...
            r.searchCost,validCost,r.final.ITSE,r.final.Jline,r.final.finalDist,r.nEval}; %#ok<AGROW>
    end
    function plotCriterion(crit)
        idx=[runs.criterion]==crit; rr=runs(idx);
        f=figure('Visible','off','Color','w'); tiledlayout(2,2);
        nexttile; plot(base.x,base.y,'k--','LineWidth',1.1); hold on;
        for q=1:numel(rr), plot(rr(q).final.x,rr(q).final.y,'LineWidth',1.2, ...
                'DisplayName',sprintf('%s semente %g',rr(q).method,rr(q).seed)); end
        plot([cfg.p0(1) cfg.pg(1)],[cfg.p0(2) cfg.pg(2)],'k:'); axis equal; grid on;
        xlabel('x (m)'); ylabel('y (m)'); title('Trajetória'); legend('Inicial','Location','best');
        nexttile; hold on; for q=1:numel(rr), plot(rr(q).final.t,rr(q).final.err,'DisplayName',sprintf('%s %g',rr(q).method,rr(q).seed)); end
        grid on; xlabel('Tempo (s)'); ylabel('Erro orientação (rad)'); title('Erro angular'); legend('show','Location','best');
        nexttile; hold on; for q=1:numel(rr), plot(rr(q).final.t,rr(q).final.dev,'DisplayName',sprintf('%s %g',rr(q).method,rr(q).seed)); end
        grid on; xlabel('Tempo (s)'); ylabel('Desvio da reta (m)'); title('Desvio'); legend('show','Location','best');
        nexttile; hold on; for q=1:numel(rr), if ~isempty(rr(q).history), plot(rr(q).history(:,6),rr(q).history(:,2),'-o','DisplayName',sprintf('%s %g',rr(q).method,rr(q).seed)); end, end
        grid on; xlabel('Avaliações acumuladas'); ylabel('Melhor custo'); title('Convergência observada'); legend('show','Location','best');
        exportgraphics(f,fullfile(cfg.outdir,sprintf('comparacao_criterio_%d.png',crit)),'Resolution',170); close(f);
    end
    function d=simulate(g)
        resetSim(); sim.setStepping(true); sim.startSimulation();
        I=0; D=0; eprev=0; A=zeros(cfg.maxSteps+1,6); reached=false; rho=inf;
        try
            for k=1:cfg.maxSteps+1
                p=vector(sim.getObjectPosition(body,sim.handle_world));
                o=vector(sim.getObjectOrientation(body,sim.handle_world));
                pg=vector(sim.getObjectPosition(goal,sim.handle_world));
                t=scalar(sim.getSimulationTime()); rho=norm(pg(1:2)-p(1:2));
                ref=atan2(pg(2)-p(2),pg(1)-p(1)); e=atan2(sin(ref-o(3)),cos(ref-o(3)));
                A(k,:)=[t p(1) p(2) o(3) e rho];
                if rho<cfg.goalTol, reached=true; break; end
                if k>cfg.maxSteps, break; end
                I=min(max(I+e*cfg.h,-cfg.iMax),cfg.iMax);
                de=atan2(sin(e-eprev),cos(e-eprev)); D=(D+g(3)*cfg.N*de)/(1+cfg.N*cfg.h);
                w=min(max(g(1)*e+g(2)*I+D,-cfg.wmax),cfg.wmax);
                wr=(2*cfg.v+w*cfg.L)/(2*cfg.R); wl=(2*cfg.v-w*cfg.L)/(2*cfg.R);
                sim.setJointTargetVelocity(left,wl); sim.setJointTargetVelocity(right,wr);
                eprev=e; sim.step();
            end
        catch ME, safeStop(sim,left,right); rethrow(ME); end
        sim.setJointTargetVelocity(left,0); sim.setJointTargetVelocity(right,0); stopAndWait();
        A=A(1:k,:); a=cfg.p0(2)-cfg.pg(2); b=cfg.pg(1)-cfg.p0(1); c=cfg.p0(1)*cfg.pg(2)-cfg.pg(1)*cfg.p0(2);
        d=struct('t',A(:,1),'x',A(:,2),'y',A(:,3),'phi',A(:,4),'err',A(:,5), ...
            'dist',A(:,6),'dev',abs(a*A(:,2)+b*A(:,3)+c)/hypot(a,b), ...
            'reached',reached,'finalDist',rho);
        d.Jline=max(d.dev); d.ITSE=trapz(d.t,d.t.*d.err.^2);
    end
    function resetSim()
        if scalar(sim.getSimulationState())~=sim.simulation_stopped, sim.stopSimulation(); end
        stopAndWait();
        % Para comparação justa, usuário deve restaurar cena via botão reset entre execuções
        % se a simulação não reiniciar objetos automaticamente.
        p=vector(sim.getObjectPosition(body,sim.handle_world));
        pg=vector(sim.getObjectPosition(goal,sim.handle_world));
        if norm(p(1:2)-cfg.p0(1:2))>1e-3 || norm(pg(1:2)-cfg.pg(1:2))>1e-3
            error('Cena fora da pose inicial. Reposicione/reset o Pioneer e execute novamente.');
        end
    end
    function stopAndWait()
        timer=tic;
        while scalar(sim.getSimulationState())~=sim.simulation_stopped
            assert(toc(timer)<10,'CoppeliaSim não parou em 10 s.'); pause(0.05);
        end
    end
    function T=poseTable(d)
        T=table(d.t,d.x,d.y,d.phi,d.err,d.dev,d.dist, ...
            'VariableNames',{'tempo_s','x_m','y_m','phi_rad','erro_rad','desvio_m','distancia_alvo_m'});
    end
end
function v=vector(v)
if iscell(v), z=zeros(1,numel(v)); for k=1:numel(v), z(k)=scalar(v{k}); end, v=z; else, v=double(v(:)'); end
end
function v=scalar(v)
while iscell(v), v=v{1}; end; v=double(v); assert(isscalar(v)&&isfinite(v),'Retorno escalar inválido da API.');
end
function safeStop(sim,left,right)
try, sim.setJointTargetVelocity(left,0); sim.setJointTargetVelocity(right,0); sim.stopSimulation(); catch, end
end
