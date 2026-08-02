% Study 1: objective ocular ICA + per-band LCMV source reconstruction, all three conditions.
%
% For each participant one ICA solution is computed from the charismatic, non-charismatic and
% silent periods concatenated, and applied to all three, so the ocular correction is identical
% across conditions within a participant. Components are classified as ocular by a fixed rule
% (frontality > 0.45, 1-4 Hz power fraction > 0.4, kurtosis > 5) with no manual selection.
% runica is seeded per participant so the decomposition is reproducible, and the same seed is
% used by ft_source_broadband_ica.m so that the band and broadband data carry the identical
% correction.
%
% Output: source_myica3/<subj>.mat -> Charismatic / Non_Charismatic / Silent = [72 x T@100Hz x 5 bands],
% plus n_rejected. Run this first; every Study-1 Python script reads its output.
%
% Reconstruction parameters: onset = sampleinfo - 1.30 s, 248 channels, per-band covariance
% LCMV regularised at 10% of the trace, Hilbert envelope, 0.8 Hz low-pass, resampled to 100 Hz.
%
% Set NSUB to limit the number of subjects when testing:  setenv('NSUB','2')
addpath('D:/fieldtrip'); ft_defaults;
global ft_default                                  % suppress FieldTrip's per-call chatter
ft_default.showcallinfo='no'; ft_default.trackcallinfo='no';
warning('off','all');
ROOT='G:/Barak1/Charisma'; OUT='G:/Barak1/reanalysis/source_myica3';
if ~exist(OUT,'dir'), mkdir(OUT); end
SFM=1017.25; FS=100; BANDS=[1 4;4 8;8 12;12 25;25 40];
CODES=[220 240 250]; CONDS={'Charismatic','Non_Charismatic','Silent'}; DUR=[150.24 140.16 153.60];
atl=load('G:/Barak1/Charisma/atlas_info.mat'); NREG=numel(atl.tissuelabel);
[blp,alp]=butter(4,0.8/(SFM/2),'low');
subs=dir(fullfile(ROOT,'char_*')); [~,o]=sort(cellfun(@(x)str2double(x(6:end)),{subs.name})); subs=subs(o);
lim=getenv('NSUB'); if ~isempty(lim), subs=subs(1:str2double(lim)); end
rej_log=nan(numel(subs),1);
ndone=numel(dir(fullfile(OUT,'char_*.mat'))); ttot=0; nrun=0;
fprintf('=== %d of %d subjects already complete ===\n',ndone,numel(subs));

for si=1:numel(subs)
  t0=tic;
  tag=subs(si).name; sd=fullfile(ROOT,tag); outf=fullfile(OUT,[tag '.mat']);
  if exist(outf,'file'), fprintf('%s skip\n',tag); continue; end
  rng(1000+si);                                  % deterministic runica initialisation
  megf=dir(fullfile(sd,'*lf_c,rfhp0.1Hz')); megf=fullfile(sd,megf(1).name);
  hm=load(fullfile(sd,'headmodelN.mat')); headmodel=ft_convert_units(hm.headmodelN,'cm');
  sm=load(fullfile(sd,'sourcemodel.mat')); sourcemodel=sm.sourcemodel;
  sourcemodel.inside=logical(sourcemodel.inside(:)); sourcemodel=ft_convert_units(sourcemodel,'cm');
  ad=load(fullfile(sd,'audiodata.mat')); ad=ad.audiodata;
  tissue=atl.tissue(:); tissue=tissue(sourcemodel.inside);

  % ---- read every available condition at full rate ----
  clip={}; have=[];
  for ci=1:numel(CODES)
    idx=find(ad.trialinfo==CODES(ci),1);
    if isempty(idx), fprintf('%s: no %s trial\n',tag,CONDS{ci}); continue; end
    onset=ad.sampleinfo(idx,1)-round(1.30*SFM); ns=round(DUR(ci)*SFM);
    cfg=[]; cfg.dataset=megf; cfg.trl=[onset onset+ns-1 0]; cfg.channel='MEG';
    clip{end+1}=ft_preprocessing(cfg); have(end+1)=ci; %#ok<SAGROW>
  end
  if numel(have)<2, fprintf('%s: fewer than 2 conditions, skipping\n',tag); continue; end

  % ---- one ICA over all available conditions ----
  rs={}; for k=1:numel(clip), cfg=[]; cfg.feedback='none'; cfg.resamplefs=300; rs{k}=ft_resampledata(cfg,clip{k}); end
  cfg=[]; cat=ft_appenddata(cfg,rs{:}); clear rs
  cfg=[]; cfg.method='runica'; cfg.numcomponent=40; cfg.runica.maxsteps=120; cfg.runica.verbose='off'; cfg.feedback='none'; comp=ft_componentanalysis(cfg,cat);
  clear cat
  [~,loc]=ismember(comp.topolabel,clip{1}.grad.label); pos=clip{1}.grad.chanpos(loc,:);
  ant=pos(:,2)>prctile(pos(:,2),70) & pos(:,3)<prctile(pos(:,3),50);
  reject=[];
  for c=1:size(comp.topo,2)
    t=comp.topo(:,c); front=norm(t(ant))/norm(t); x=comp.trial{1}(c,:); k=kurtosis(x);
    P=abs(fft(x-mean(x))).^2; f=(0:numel(x)-1)/numel(x)*300; lfrac=sum(P(f>=1&f<4))/sum(P(f>=1&f<40));
    if front>0.45 && lfrac>0.4 && k>5, reject(end+1)=c; end %#ok<SAGROW>
  end
  rej_log(si)=numel(reject); fprintf('%s: reject %d comps %s\n',tag,numel(reject),mat2str(reject));

  % ---- clean and beamform each condition with that one solution ----
  lf=[]; reg=struct();
  for k=1:numel(clip)
    ci=have(k);
    X=clip{k}.trial{1}; [~,order]=ismember(clip{k}.label,comp.topolabel);
    topo=comp.topo(order,:); unmix=comp.unmixing(:,order);
    if isempty(reject), Xc=X; else Xd=X-mean(X,2); S=unmix*Xd; Xc=X-topo(:,reject)*S(reject,:); end
    cln=clip{k}; cln.trial{1}=Xc;
    if isempty(lf), cfg=[]; cfg.sourcemodel=sourcemodel; cfg.headmodel=headmodel; cfg.channel='MEG'; cfg.feedback='none';
      lf=ft_prepare_leadfield(cfg,cln); end
    ns=size(cln.trial{1},2); T100=round(DUR(ci)*FS); tv=(0:T100-1)/FS; tsrc=(0:ns-1)/SFM;
    R=zeros(NREG,T100,size(BANDS,1),'single');
    for bi=1:size(BANDS,1)
      cfg=[]; cfg.bpfilter='yes'; cfg.bpfreq=BANDS(bi,:); cfg.bpfilttype='firws'; cfg.feedback='none'; db=ft_preprocessing(cfg,cln);
      cfg=[]; cfg.covariance='yes'; cfg.covariancewindow='all'; tl=ft_timelockanalysis(cfg,db);
      cfg=[]; cfg.method='lcmv'; cfg.sourcemodel=lf; cfg.headmodel=headmodel;
      cfg.lcmv.lambda='10%'; cfg.lcmv.keepfilter='yes'; cfg.lcmv.fixedori='yes'; cfg.lcmv.feedback='none'; src=ft_sourceanalysis(cfg,tl);
      ins=find(sourcemodel.inside); flt=src.avg.filter; F=zeros(numel(ins),size(db.trial{1},1));
      for kk=1:numel(ins), f=flt{ins(kk)}; if ~isempty(f), F(kk,:)=f; end, end
      vts=F*db.trial{1};
      for r=1:NREG, v=vts(tissue==r,:); if isempty(v),continue; end
        env=abs(hilbert(mean(v,1)')); env=filtfilt(blp,alp,env); R(r,:,bi)=single(interp1(tsrc,env,tv,'linear','extrap')); end
    end
    reg.(CONDS{ci})=R;
  end
  reg.n_rejected=numel(reject);
  save(outf,'-struct','reg','-v7');
  el=toc(t0)/60; nrun=nrun+1; ttot=ttot+el; ndone=ndone+1;
  left=(numel(subs)-ndone)*(ttot/nrun);
  fprintf('[%2d/%d] %s  %.1f min  | this session %.0f min, ETA %.0f min (%.1f h)\n', ...
          ndone,numel(subs),tag,el,ttot,left,left/60);
end
fprintf('reject counts: %s | mean=%.1f\n',mat2str(rej_log'),nanmean(rej_log));
fprintf('ALL DONE\n');
