% Studies 2 & 3 (ISCex) source reconstruction — objective ICA + LCMV beamforming.
% Input: G:/Barak1/iscex/ISCex_N/dataorig.mat (pre-ICA, 3 trials: rest / S2 video / S3 video),
% 245 MEG ch, fs=1017.25. Same objective blink-ICA and per-band-covariance LCMV as Study 1
% (ft_ica_full3.m), adapted to the segmented dataorig input and this cohort's headmodelN /
% sourcemodelN (6426-node template grid matching atlas_info.tissue; 1169 AAL-labelled voxels).
% Per subject, per trial (rest,S2,S3) produces:
%   *_band : 72 x T@100Hz x 5   per-band Hilbert-envelope (per-band covariance) -> ISC
%   *_bb   : 72 x T@250Hz       broadband (1-40Hz) region series               -> specparam
% Output: G:/Barak1/iscex/source_iscex/ISCex_N.mat  (skips existing; resumable).
addpath('D:/fieldtrip'); ft_defaults;
ROOT='G:/Barak1/iscex'; OUT='G:/Barak1/iscex/source_iscex';
if ~exist(OUT,'dir'), mkdir(OUT); end
SFM=1017.25; FS=100; FSD=250; BANDS=[1 4;4 8;8 12;12 25;25 40];
SKIP=[4 12 25 29];                                   % too noisy -> drop entirely
atl=load(fullfile(ROOT,'ISCex_1','atlas_info.mat')); NREG=numel(atl.tissuelabel);
[blp,alp]=butter(4,0.8/(SFM/2),'low');
d=dir(fullfile(ROOT,'ISCex_*')); d=d([d.isdir]);
ids=cellfun(@(x)str2double(x(7:end)),{d.name}); [ids,o]=sort(ids); d=d(o);
onesub=getenv('ONESUB');

for si=1:numel(d)
  id=ids(si); tag=d(si).name; sd=fullfile(ROOT,tag); outf=fullfile(OUT,[tag '.mat']);
  if any(id==SKIP), continue; end
  if ~exist(fullfile(sd,'dataorig.mat'),'file'), continue; end
  if ~isempty(onesub) && id~=str2double(onesub), continue; end
  np=getenv('NPART'); if ~isempty(np) && mod(id,str2double(np))~=str2double(getenv('PART')), continue; end
  if exist(outf,'file'), fprintf('%s skip\n',tag); continue; end
  t0=tic;
  L=load(fullfile(sd,'dataorig.mat')); do=L.dataorig; clear L;
  hm=load(fullfile(sd,'headmodelN.mat')); headmodel=ft_convert_units(hm.headmodelN,'cm');
  sm=load(fullfile(sd,'sourcemodelN.mat')); sourcemodel=sm.sourcemodelN;
  sourcemodel.inside=logical(atl.tissue(:)>0);        % restrict to 1169 AAL-labelled voxels
  sourcemodel=ft_convert_units(sourcemodel,'cm');
  tissue=atl.tissue(:); tissue=tissue(sourcemodel.inside);
  % keep MEG channels only
  cfg=[]; cfg.channel='MEG'; do=ft_selectdata(cfg,do);
  % map trials by trigger code: rest {80,90}, S2 {200,250}, S3 {100,150}
  ti=do.trialinfo(:); ri=find(ismember(ti,[80 90]),1); i2=find(ismember(ti,[200 250]),1); i3=find(ismember(ti,[100 150]),1);
  order=[ri i2 i3]; names={'rest','S2','S3'};
  % ICA on S2+S3 videos, 300 Hz
  sub=do; sub.trial=do.trial(order(2:3)); sub.time=do.time(order(2:3));
  cfg=[]; cfg.resamplefs=300; rs=ft_resampledata(cfg,sub);
  cfg=[]; cfg.method='runica'; cfg.numcomponent=40; cfg.runica.maxsteps=120; comp=ft_componentanalysis(cfg,rs);
  % objective blink rule (frontal topo, 1-4Hz dominant, high kurtosis)
  [~,loc]=ismember(comp.topolabel,do.grad.label); pos=do.grad.chanpos(loc,:);
  ant=pos(:,2)>prctile(pos(:,2),70) & pos(:,3)<prctile(pos(:,3),50);
  reject=[];
  for c=1:size(comp.topo,2)
    t=comp.topo(:,c); front=norm(t(ant))/norm(t); x=comp.trial{1}(c,:); k=kurtosis(x);
    P=abs(fft(x-mean(x))).^2; f=(0:numel(x)-1)/numel(x)*300; lfrac=sum(P(f>=1&f<4))/sum(P(f>=1&f<40));
    if front>0.45 && lfrac>0.4 && k>5, reject(end+1)=c; end %#ok<AGROW>
  end
  fprintf('%s: reject %d comps %s\n',tag,numel(reject),mat2str(reject));
  [~,ord2]=ismember(do.label,comp.topolabel); topo=comp.topo(ord2,:); unmix=comp.unmixing(:,ord2);
  reg=struct(); reg.reject=numel(reject); reg.attitude=ti(order(2)); reg.cue=ti(order(3));
  lf=[];
  for k=1:3
    tr=order(k); X=do.trial{tr};
    if isempty(reject), Xc=X; else Xd=X-mean(X,2); S=unmix*Xd; Xc=X-topo(:,reject)*S(reject,:); end
    cln=do; cln.trial={Xc}; cln.time=do.time(tr); cln.sampleinfo=do.sampleinfo(tr,:);
    ns=size(Xc,2);
    if isempty(lf), cfg=[]; cfg.sourcemodel=sourcemodel; cfg.headmodel=headmodel; cfg.channel='MEG'; lf=ft_prepare_leadfield(cfg,cln); end
    % ---- per-band envelopes (per-band covariance) ----
    T100=round(ns/SFM*FS); tv=(0:T100-1)/FS; tsrc=(0:ns-1)/SFM;
    Rb=zeros(NREG,T100,size(BANDS,1),'single');
    for bi=1:size(BANDS,1)
      cfg=[]; cfg.bpfilter='yes'; cfg.bpfreq=BANDS(bi,:); cfg.bpfilttype='firws'; db=ft_preprocessing(cfg,cln);
      cfg=[]; cfg.covariance='yes'; cfg.covariancewindow='all'; tl=ft_timelockanalysis(cfg,db);
      cfg=[]; cfg.method='lcmv'; cfg.sourcemodel=lf; cfg.headmodel=headmodel;
      cfg.lcmv.lambda='10%'; cfg.lcmv.keepfilter='yes'; cfg.lcmv.fixedori='yes'; src=ft_sourceanalysis(cfg,tl);
      F=cat(1,src.avg.filter{sourcemodel.inside}); vts=F*db.trial{1};
      for r=1:NREG, v=vts(tissue==r,:); if isempty(v),continue; end
        env=abs(hilbert(mean(v,1)')); env=filtfilt(blp,alp,env);
        Rb(r,:,bi)=single(interp1(tsrc,env,tv,'linear','extrap')); end
    end
    reg.([names{k} '_band'])=Rb;
    % ---- broadband 250 Hz region series (for specparam) ----
    cfg=[]; cfg.bpfilter='yes'; cfg.bpfreq=[1 40]; cfg.bpfilttype='firws'; db=ft_preprocessing(cfg,cln);
    cfg=[]; cfg.covariance='yes'; cfg.covariancewindow='all'; tl=ft_timelockanalysis(cfg,db);
    cfg=[]; cfg.method='lcmv'; cfg.sourcemodel=lf; cfg.headmodel=headmodel;
    cfg.lcmv.lambda='10%'; cfg.lcmv.keepfilter='yes'; cfg.lcmv.fixedori='yes'; src=ft_sourceanalysis(cfg,tl);
    F=cat(1,src.avg.filter{sourcemodel.inside}); vts=F*db.trial{1};
    Rr=zeros(NREG,ns,'single'); for r=1:NREG, v=vts(tissue==r,:); if ~isempty(v), Rr(r,:)=single(mean(v,1)); end, end
    reg.([names{k} '_bb'])=single(resample(double(Rr'),FSD,round(SFM))');
    fprintf('  %s %s done\n',tag,names{k});
  end
  save(outf,'-struct','reg','-v7'); fprintf('%s SAVED (%.1f min)\n',tag,toc(t0)/60);
end
fprintf('ALL DONE\n');
