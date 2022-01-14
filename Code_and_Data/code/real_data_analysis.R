### real data embeddings ###

######################### load required packages ###############################

library(phyloseq)
library(adaptiveGPCA)
library(ggplot2)
library(vegan)
library(ade4)
library(MatrixCorrelation)
library(ggpubr)
library(Matrix)

######################### end load packages ####################################

######################### load real data #######################################

load("../data/phyloseq_objects.RData")

######################### end load real data ###################################


############################# Functions ########################################

#' Function to filtering out rare RSVs
#' 
#' @param phylo_obj a phyloseq object
#' @param count minimum number of occurance for a RSV used in filtering
#' @param portion used in filtering
#' @param psudo_count number to be added when 0 is prsent
#' 
#' @return a phyloseq object with pruned phylogeny 
filter_rare <- function(phylo_obj, count, portion, psudo_cnt = 0) {
  require(phyloseq)
  res = phyloseq::filter_taxa(phylo_obj, function(x) sum(x > count) > 
                                (portion*length(x)), TRUE)
  if (psudo_cnt) {
    res = phyloseq::transform_sample_counts(res, function(x) x+psudo_cnt)}
  return(res)
}

#' Function to return the M matrix based on sequence expression and k-mer
#' 
#' @param abu_tab an matrix of the raw abundance table
#' @param k k in k-mer analysis
#' 
#' @return a k-mer counting matrix m 
M_mat <- function(abu_tab, k) {
  require(seqTools)
  n_row = nrow(abu_tab)
  n_col = ncol(abu_tab)
  m = matrix(NaN, n_col, 4^k)
  for (i in seq(n_col)) {
    m[i,] = seqTools::countDnaKmers(colnames(abu_tab)[i], k=k)
  }
  return(m)
}

#' Function doing the clr transformation
#' 
#' @param abu_mat the raw abundance matrix
#' 
#' @return the clr transformed data matrix
clr_transform <- function(abu_mat){
  abu_copy = abu_mat
  for (i in c(1:nrow(abu_mat))) {
    for (j in c(1:ncol(abu_mat))) {
      abu_copy[i,j] = log(abu_mat[i,j] / exp(mean(log(abu_mat[i,]))))
    }
  }
  return(abu_copy)
}

#' Function to do the gPCA
#' 
#' @param raw_X the raw abundance matrix or transformed abundance matrix
#' @param M the k-mer counting matrix
#' @param clr_transformed if TRUE, the raw abundance matrix is clr-transformed
#' 
#' @return a list of new coordinates and ratio of variance explained
gPCA <- function(raw_X, M, nf, clr_transformed = FALSE) {
  if (!clr_transformed) {
    w_samples <- rowSums(raw_X)/sum(raw_X)
    w_esp <- colSums(raw_X)/sum(raw_X)
    X_ab = sweep(raw_X, 1, rowSums(raw_X), "/")
    X_ab = X_ab %*% (diag(x=rep(1,ncol(raw_X))) - 
                       matrix(data = rep(1,ncol(raw_X)), ncol = 1) %*% t(w_esp))
    
    trans_X = X_ab %*% M
    eigdcomp <- eigen(t(trans_X) %*% diag(x=w_samples) %*% trans_X)
    eigvals <- eigdcomp$values
    eigvecs <- eigdcomp$vectors
    
    return(list(li = as.data.frame(t(t(eigvecs[,1:nf]) %*% t(trans_X))), 
                p = eigvals[1:nf] / sum(eigvals)))}
  
  else {
    trans_X = raw_X %*% M
    eigdcomp <- eigen(t(trans_X) %*% trans_X)
    eigvals <- eigdcomp$values
    eigvecs <- eigdcomp$vectors
    
    return(list(li = as.data.frame(trans_X %*% eigvecs[,1:nf]), 
                p = eigvals[1:nf] / sum(eigvals)))
  }
}

#' Function to do the dpcoa with a distance matrix and clr transformed data
#' 
#' @param x_clr the clr transformed data matrix
#' @param dist_mat the (squared) distance matrix used in MDS in dpcoa
#' @param nf number of kept axes
#' 
#' @return a list of new coordinates and ratio of variance explained
dpcoa_clr <- function(x_clr, dist_mat, nf) {
  w_taxa = rep(1 / nrow(dist_mat), nrow(dist_mat))
  D_w_taxa = diag(x = sqrt(w_taxa))
  P_w_taxa = diag(x=rep(1,nrow(dist_mat))) - 
    matrix(data = rep(1,nrow(dist_mat)), ncol = 1) %*% t(w_taxa)
  #S = -0.5 * D_w_taxa %*% P_w_taxa %*% dist_mat %*% t(P_w_taxa) %*% D_w_taxa
  S = -0.5 * P_w_taxa %*% dist_mat %*% t(P_w_taxa)
  eig = eigen(S)
  #Z = diag(x = 1/sqrt(w_taxa)) %*% eig$vectors[,1:(nrow(dist_mat)-1)] %*% diag(x = sqrt(eig$values[1:(nrow(dist_mat)-1)]))
  Z = eig$vectors[,1:(nrow(dist_mat)-1)] %*% 
    diag(x = sqrt(eig$values[1:(nrow(dist_mat)-1)]))
  Y = x_clr %*% Z
  ordi = adaptiveGPCA::gpca(Y, diag(x=rep(1,nrow(dist_mat)-1)), k = nf)
  return(list(li = Y %*% ordi$V, eig = ordi$vars[1:nf]))
  # ordi = gPCA2(Y, diag(x=rep(1,nrow(dist_mat)-1)), 2)
}

#' Function to do the dpcoa based on the raw data
#' 
#' @param x_raw the raw abundance matrix
#' @param dist_mat the (squared) distance matrix used in MDS in dpcoa
#' @param nf number of kept axes
#' 
#' @return a list of new coordinates and ratio of variance explained
dpcoa_raw <- function(x_raw, dist_mat, nf) {
  w_samples <- rowSums(x_raw)/sum(x_raw)
  w_esp <- colSums(x_raw)/sum(x_raw)
  X_ab = sweep(x_raw, 1, rowSums(x_raw), "/")
  D_w_taxa = diag(x = sqrt(w_esp))
  P_w_taxa = diag(x=rep(1,nrow(dist_mat))) - 
    matrix(data = rep(1,nrow(dist_mat)), ncol = 1) %*% t(w_esp)
  S = -0.5 * D_w_taxa %*% P_w_taxa %*% dist_mat %*% t(P_w_taxa) %*% D_w_taxa
  eig = eigen(S)
  Z = diag(x = 1 / sqrt(w_esp)) %*% eig$vectors[,1:(nrow(dist_mat)-1)] %*% 
    diag(x = sqrt(eig$values[1:(nrow(dist_mat)-1)]))
  Y = X_ab %*% Z
  ordi = adaptiveGPCA::gpca(Y,diag(x=rep(1,nrow(dist_mat)-1)), w_samples, k = nf)
  return(list(li = Y %*% ordi$V, eig = ordi$vars[1:nf]))
}

#' Function to do the procrustean analysis
#' 
#' @param benchmark benchmark embedding results
#' @param alternative embedding result to compare
#' 
#' @return the original embeddings X given by benchmark and the rotated
#' embeddings Y of alternative obtained by rotation
proc_X_Y <- function(benchmark, alternative) {
  require(vegan)
  pro = vegan::procrustes(benchmark, alternative)
  return(list(X = pro$X, Y = pro$Yrot))
}

#' Function to do RV coef calculation
#' 
#' @param proc an list returned by function proc_X_Y
#' 
#' @return the RV coef based on procrustean analysis
rv <- function(p) {
  require(MatrixCorrelation)
  return(MatrixCorrelation::RV(p$X, p$Y))
}
########################### end functions ######################################


########################### Data analysis ######################################



########################### Analysis in the supplementary ######################

# different levels of pruning
thresh = c(0.01, 0.05, 0.1, 0.15, 0.2, 0.25)

df_raw <- data.frame(rv=double(),
                     method=character(), 
                     thresh=double(), 
                     stringsAsFactors=FALSE)
df_clr <- data.frame(rv=double(),
                     method=character(), 
                     thresh=double(), 
                     stringsAsFactors=FALSE) 
for (i in thresh) {
  pruned_phylo = filter_rare(cc, 3, i)
  pruned_phylo_clr = filter_rare(cc, 3, i, 1)
  # squared distance matrix for DPCoA with clr transformed data
  dist_mat=as.matrix((as.dist(ape::cophenetic.phylo(phy_tree(pruned_phylo_clr)))))
  
  # dimension of the data matrix
  n = nrow(phyloseq::otu_table(pruned_phylo))
  p = ncol(phyloseq::otu_table(pruned_phylo))
  
  # raw data matrix and clr transformed data matrix
  X_raw = as(otu_table(pruned_phylo), "matrix")
  X_clr = as(otu_table(pruned_phylo_clr), "matrix")
  X_clr = clr_transform(as(otu_table(pruned_phylo_clr), "matrix"))
  
  # k-mer counting matrix M for different values of k
  M1 = M_mat(X_raw, 1)
  M2 = M_mat(X_raw, 2)
  M3 = M_mat(X_raw, 3)
  M4 = M_mat(X_raw, 4)
  M5 = M_mat(X_raw, 5)
  M6 = M_mat(X_raw, 6)
  
  # DPCoA based on raw abundance matrix
  dp_raw = dpcoa_raw(X_raw, dist_mat, nf = min(nrow(X_raw), ncol(X_raw)) - 1)
  # gPCA based on raw abundance matrix for various k
  gpca_raw_1 = gPCA(X_raw, M1, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M1 %*% t(M1))) - 1)
  gpca_raw_2 = gPCA(X_raw, M2, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M2 %*% t(M2))) - 1)
  gpca_raw_3 = gPCA(X_raw, M3, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M3 %*% t(M3))) - 1)
  gpca_raw_4 = gPCA(X_raw, M4, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M4 %*% t(M4))) - 1)
  gpca_raw_5 = gPCA(X_raw, M5, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M5 %*% t(M5))) - 1)
  gpca_raw_6 = gPCA(X_raw, M6, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M6 %*% t(M6))) - 1)
  # Standard PCA on raw abundance matrix
  pca_raw = gPCA(X_raw, diag(x=rep(1,p)), 
                 nf = min(nrow(X_raw), ncol(X_raw)) - 1)

  
  # DPCoA based on clr transformed matrix
  dp_clr = dpcoa_clr(X_clr, dist_mat, nf = min(nrow(X_raw), ncol(X_raw)) - 1)
  
  # gPCA based on clr transformed abundance matrix for various k
  gpca_clr_1 = gPCA(X_clr, M1, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M1 %*% t(M1))) - 1, 
                    clr_transformed = TRUE)
  gpca_clr_2 = gPCA(X_clr, M2, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M2 %*% t(M2))) - 1, 
                    clr_transformed = TRUE)
  gpca_clr_3 = gPCA(X_clr, M3, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M3 %*% t(M3))) - 1, 
                    clr_transformed = TRUE)
  gpca_clr_4 = gPCA(X_clr, M4, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M4 %*% t(M4))) - 1, 
                    clr_transformed = TRUE)
  gpca_clr_5 = gPCA(X_clr, M5, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M5 %*% t(M5))) - 1, 
                    clr_transformed = TRUE)
  gpca_clr_6 = gPCA(X_clr, M6, nf = 
                      min(nrow(X_raw), 
                          ncol(X_raw), rankMatrix(M6 %*% t(M6))) - 1, 
                    clr_transformed = TRUE)
  # Standard PCA on clr transformed abundance matrix
  pca_clr = gPCA(X_clr, diag(x=rep(1,p)), nf = 
                   min(nrow(X_raw), ncol(X_raw)) - 1, clr_transformed = TRUE)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, pca_raw$li)), 'pca', i)
  # RV for gpca
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_1$li)), 'k=1', i)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_2$li)), 'k=2', i)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_3$li)), 'k=3', i)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_4$li)), 'k=4', i)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_5$li)), 'k=5', i)
  df_raw[nrow(df_raw)+1,] = c(rv(proc_X_Y(dp_raw$li, gpca_raw_6$li)), 'k=6', i)
  
  # RV w.r.t the clr data
  
  # RV for pca
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, pca_clr$li)), 'pca', i)
  # RV for gpca
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_1$li)), 'k=1', i)
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_2$li)), 'k=2', i)
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_3$li)), 'k=3', i)
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_4$li)), 'k=4', i)
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_5$li)), 'k=5', i)
  df_clr[nrow(df_clr)+1,] = c(rv(proc_X_Y(dp_clr$li, gpca_clr_6$li)), 'k=6', i)
}


for (i in seq(42)) {
  df_raw[i,]$rv = as.numeric(round(as.numeric(df_raw[i,]$rv), 3))
  df_clr[i,]$rv = as.numeric(round(as.numeric(df_clr[i,]$rv), 3))
}
df_raw[, 1] <- sapply(df_raw[, 1], as.numeric)
df_clr[, 1] <- sapply(df_clr[, 1], as.numeric)
p1 <- ggplot2::ggplot(df_raw, aes(thresh, (rv), 
                                  group = method, color = method)) + 
  geom_point() + geom_line() + cowplot::theme_half_open() + 
  coord_cartesian(ylim = c(0.15, 1)) +
  cowplot::background_grid() + xlab("level of pruning") + 
  ylab("RV coefficient") + 
  ggtitle("Relative Abundance Transformation")
p2 <- ggplot2::ggplot(df_clr, aes(thresh, (rv), 
                                  group = method, color = method)) + 
  geom_point() + geom_line() + cowplot::theme_half_open() + 
  coord_cartesian(ylim = c(0.15, 1)) +
  cowplot::background_grid() + xlab("level of pruning") + 
  ylab("RV coefficient") + 
  ggtitle("Centered Log-ratio Transformation")
ggpubr::ggarrange(p1, p2, ncol=2, nrow=1, common.legend = TRUE, legend = 'right')
################################################################################


####################### analysis in the paper ##################################

############### analysis based on raw abundance matrix #########################

pruned_phylo = filter_rare(cc, 3, 0.2)
pruned_phylo_clr = filter_rare(cc, 3, 0.2, 1)
# squared distance matrix for DPCoA with clr transformed data
dist_mat=as.matrix((as.dist(ape::cophenetic.phylo(phy_tree(pruned_phylo_clr)))))

# dimension of the data matrix
n = nrow(phyloseq::otu_table(pruned_phylo))
p = ncol(phyloseq::otu_table(pruned_phylo))

# raw data matrix and clr transformed data matrix
X_raw = as(otu_table(pruned_phylo), "matrix")
X_clr = as(otu_table(pruned_phylo_clr), "matrix")
X_clr = clr_transform(as(otu_table(pruned_phylo_clr), "matrix"))

# k-mer counting matrix M for different values of k
M1 = M_mat(X_raw, 1)
M2 = M_mat(X_raw, 2)
M3 = M_mat(X_raw, 3)
M4 = M_mat(X_raw, 4)
M5 = M_mat(X_raw, 5)
M6 = M_mat(X_raw, 6)

# DPCoA based on raw abundance matrix
dp_raw = dpcoa_raw(X_raw, dist_mat, nf = min(nrow(X_raw), ncol(X_raw)) - 1)
# gPCA based on raw abundance matrix for various k
gpca_raw_1 = gPCA(X_raw, M1, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M1 %*% t(M1))) - 1)
gpca_raw_2 = gPCA(X_raw, M2, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M2 %*% t(M2))) - 1)
gpca_raw_3 = gPCA(X_raw, M3, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M3 %*% t(M3))) - 1)
gpca_raw_4 = gPCA(X_raw, M4, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M4 %*% t(M4))) - 1)
gpca_raw_5 = gPCA(X_raw, M5, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M5 %*% t(M5))) - 1)
gpca_raw_6 = gPCA(X_raw, M6, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M6 %*% t(M6))) - 1)
# Standard PCA on raw abundance matrix
pca_raw = gPCA(X_raw, diag(x=rep(1,p)), nf = min(nrow(X_raw), ncol(X_raw)) - 1)

################ analysis based on clr transformed matrix ######################
# DPCoA based on clr transformed matrix
dp_clr = dpcoa_clr(X_clr, dist_mat, nf = min(nrow(X_raw), ncol(X_raw)) - 1)

# gPCA based on clr transformed abundance matrix for various k
gpca_clr_1 = gPCA(X_clr, M1, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M1 %*% t(M1))) - 1, 
                  clr_transformed = TRUE)
gpca_clr_2 = gPCA(X_clr, M2, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M2 %*% t(M2))) - 1, 
                  clr_transformed = TRUE)
gpca_clr_3 = gPCA(X_clr, M3, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M3 %*% t(M3))) - 1, 
                  clr_transformed = TRUE)
gpca_clr_4 = gPCA(X_clr, M4, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M4 %*% t(M4))) - 1, 
                  clr_transformed = TRUE)
gpca_clr_5 = gPCA(X_clr, M5, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M5 %*% t(M5))) - 1, 
                  clr_transformed = TRUE)
gpca_clr_6 = gPCA(X_clr, M6, nf = 
                    min(nrow(X_raw), 
                        ncol(X_raw), rankMatrix(M6 %*% t(M6))) - 1, 
                  clr_transformed = TRUE)
# Standard PCA on clr transformed abundance matrix
pca_clr = gPCA(X_clr, diag(x=rep(1,p)), nf = 
                 min(nrow(X_raw), ncol(X_raw)) - 1, clr_transformed = TRUE)

######################## RV coef ###############################################

# RV w.r.t the raw data

# RV for pca
rv(proc_X_Y(dp_raw$li, pca_raw$li))
# RV for gpca
rv(proc_X_Y(dp_raw$li, gpca_raw_1$li))
rv(proc_X_Y(dp_raw$li, gpca_raw_2$li))
rv(proc_X_Y(dp_raw$li, gpca_raw_3$li))
rv(proc_X_Y(dp_raw$li, gpca_raw_4$li))
rv(proc_X_Y(dp_raw$li, gpca_raw_5$li))
rv(proc_X_Y(dp_raw$li, gpca_raw_6$li))

# RV w.r.t the clr data

# RV for pca
rv(proc_X_Y(dp_clr$li, pca_clr$li))
# RV for gpca
rv(proc_X_Y(dp_clr$li, gpca_clr_1$li))
rv(proc_X_Y(dp_clr$li, gpca_clr_2$li))
rv(proc_X_Y(dp_clr$li, gpca_clr_3$li))
rv(proc_X_Y(dp_clr$li, gpca_clr_4$li))
rv(proc_X_Y(dp_clr$li, gpca_clr_5$li))
rv(proc_X_Y(dp_clr$li, gpca_clr_6$li))
############################# end data analysis ################################




############################# Some plots #######################################

######## plots regarding analysis of raw abundance matrix ######################

# Procrustean analysis for each method compared with the benchmark (DPCoA)
# add columns considering the meta data
embed_dp_raw = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_1$li)$X)
embed_dp_raw$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_1 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_1$li)$Y)
embed_dp_raw_1$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_1$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_2 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_2$li)$Y)
embed_dp_raw_2$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_2$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_3 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_3$li)$Y)
embed_dp_raw_3$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_3$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_4 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_4$li)$Y)
embed_dp_raw_4$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_4$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_5 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_5$li)$Y)
embed_dp_raw_5$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_5$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_raw_6 = as.data.frame(proc_X_Y(dp_raw$li, gpca_raw_6$li)$Y)
embed_dp_raw_6$Subject = sample_data(pruned_phylo)$Subject
embed_dp_raw_6$Time = sample_data(pruned_phylo)$CC_Interval
embed_pca_raw = as.data.frame(proc_X_Y(dp_raw$li, pca_raw$li)$Y)
embed_pca_raw$Subject = sample_data(pruned_phylo)$Subject
embed_pca_raw$Time = sample_data(pruned_phylo)$CC_Interval

# facet plots
p_dp_raw <- ggplot2::ggplot(embed_dp_raw, 
                            aes(Axis1, Axis2, 
                                color = Subject, group = Time, shape = Time)) +
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(dp_raw$eig[1] / sum(dp_raw$eig),3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(dp_raw$eig[2] / sum(dp_raw$eig),3)*100, "%")) +
  ggtitle("DPCoA") + facet_wrap(~Subject)

p_pca_raw <- ggplot2::ggplot(embed_pca_raw, 
                             aes(V1, V2, 
                                 color = Subject, group = Time, shape = Time)) +
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(pca_raw$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(pca_raw$p[2],3)*100, "%")) +
  ggtitle("PCA") + facet_wrap(~Subject)

p_dp_raw_2 <- ggplot2::ggplot(embed_dp_raw_2, 
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_raw_2$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_raw_2$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=2") + facet_wrap(~Subject)

p_dp_raw_3 <- ggplot2::ggplot(embed_dp_raw_3,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_raw_3$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_raw_3$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=3") + facet_wrap(~Subject)

p_dp_raw_4 <- ggplot2::ggplot(embed_dp_raw_4,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_raw_4$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_raw_4$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=4") + facet_wrap(~Subject)

p_dp_raw_5 <- ggplot2::ggplot(embed_dp_raw_5,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_raw_5$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_raw_5$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=5") + facet_wrap(~Subject)

ggpubr::ggarrange(p_dp_raw, p_pca_raw, p_dp_raw_2, 
                  p_dp_raw_3, p_dp_raw_4, 
                  p_dp_raw_5, ncol=3, nrow=2, common.legend = TRUE, 
                  legend = 'right')

######## plots regarding the clr transformed data matrix #######################

# Procrustean analysis for each method compared with the benchmark (DPCoA)
# add columns considering the meta data
embed_dp_clr = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_1$li)$X)
embed_dp_clr$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_1 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_1$li)$Y)
embed_dp_clr_1$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_1$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_2 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_2$li)$Y)
embed_dp_clr_2$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_2$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_3 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_3$li)$Y)
embed_dp_clr_3$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_3$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_4 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_4$li)$Y)
embed_dp_clr_4$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_4$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_5 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_5$li)$Y)
embed_dp_clr_5$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_5$Time = sample_data(pruned_phylo)$CC_Interval
embed_dp_clr_6 = as.data.frame(proc_X_Y(dp_clr$li, gpca_clr_6$li)$Y)
embed_dp_clr_6$Subject = sample_data(pruned_phylo)$Subject
embed_dp_clr_6$Time = sample_data(pruned_phylo)$CC_Interval
embed_pca_clr = as.data.frame(proc_X_Y(dp_clr$li, pca_clr$li)$Y)
embed_pca_clr$Subject = sample_data(pruned_phylo)$Subject
embed_pca_clr$Time = sample_data(pruned_phylo)$CC_Interval

# facet plots
p_dp_clr <- ggplot2::ggplot(embed_dp_clr, 
                            aes(Axis1, Axis2, 
                                color = Subject, group = Time, shape = Time)) +
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(dp_clr$eig[1] / sum(dp_clr$eig),3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(dp_clr$eig[2] / sum(dp_clr$eig),3)*100, "%")) +
  ggtitle("DPCoA") + facet_wrap(~Subject)

p_pca_clr <- ggplot2::ggplot(embed_pca_clr, 
                             aes(V1, V2, 
                                 color = Subject, group = Time, shape = Time)) +
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(pca_clr$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(pca_clr$p[2],3)*100, "%")) +
  ggtitle("PCA") + facet_wrap(~Subject)

p_dp_clr_2 <- ggplot2::ggplot(embed_dp_clr_2, 
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_clr_2$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_clr_2$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=2") + facet_wrap(~Subject)

p_dp_clr_3 <- ggplot2::ggplot(embed_dp_clr_3,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_clr_3$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_clr_3$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=3") + facet_wrap(~Subject)

p_dp_clr_4 <- ggplot2::ggplot(embed_dp_clr_4,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_clr_4$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_clr_4$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=4") + facet_wrap(~Subject)

p_dp_clr_5 <- ggplot2::ggplot(embed_dp_clr_5,
                              aes(V1, V2, 
                                  color = Subject, group = Time, shape = Time))+
  geom_point() + cowplot::theme_half_open() + cowplot::background_grid() +
  xlab(paste0("Axis 1: ", round(gpca_clr_5$p[1],3)*100, "%")) +
  ylab(paste0("Axis 2: ", round(gpca_clr_5$p[2],3)*100, "%")) +
  ggtitle("gPCA, k=5") + facet_wrap(~Subject)

ggpubr::ggarrange(p_dp_clr, p_pca_clr, p_dp_clr_2, 
                  p_dp_clr_3, p_dp_clr_4, 
                  p_dp_clr_5, ncol=3, nrow=2, common.legend = TRUE, 
                  legend = 'right')

# Q_p eigenstructure plot for DPCoA on different trees

bt = ape::read.tree("../data/btree.tree")
dist_mat = as.matrix(as.dist(ape::cophenetic.phylo(bt)))
# ade4::is.euclid(sqrt(as.dist(ape::cophenetic.phylo(t)))
p = 32
# standard double centering
w_taxa = rep(1 / nrow(dist_mat), nrow(dist_mat))
P_w_taxa = diag(x=rep(1,nrow(dist_mat))) - 
  matrix(data = rep(1,nrow(dist_mat)), ncol = 1) %*% t(w_taxa)
S = -0.5 * P_w_taxa %*% dist_mat %*% t(P_w_taxa)
decomp = eigen(S, TRUE)

df_bt <- data.frame(x=rep(1:32, 3), val=c(decomp$vectors[,1], decomp$vectors[,2],
                                          decomp$vectors[,3]), 
                    variable=rep(paste0("eigenvector ", 1:3), each=32), 
                    tree_type=rep("balanced binary tree", 96))

ct = ape::read.tree("../data/ctree.tree")
dist_mat = as.matrix(as.dist(ape::cophenetic.phylo(ct)))
ade4::is.euclid(sqrt(as.dist(ape::cophenetic.phylo(ct))))
# standard double centering
w_taxa = rep(1 / nrow(dist_mat), nrow(dist_mat))
P_w_taxa = diag(x=rep(1,nrow(dist_mat))) - 
  matrix(data = rep(1,nrow(dist_mat)), ncol = 1) %*% t(w_taxa)
S = -0.5 * P_w_taxa %*% dist_mat %*% t(P_w_taxa)
decomp = eigen(S, TRUE)

df_ct <- data.frame(x=rep(1:32, 3), val=c(decomp$vectors[,1], 
                                          decomp$vectors[,2],
                                          decomp$vectors[,3]), 
                    variable=rep(paste0("eigenvector ", 1:3), each=32), 
                    tree_type=rep("comb tree", 96))

rt = ape::read.tree("../data/real_tree.xml")

index_order = c(5, 4, 3, 1, 2,
                50, 48, 49, 45, 46,
                47, 44, 38, 36, 37,
                42, 43, 39, 40, 41,
                35, 34, 28, 29, 30,
                31, 32, 33, 27, 26,
                24, 25, 23, 18, 19,
                22, 20, 21, 17, 15,
                16, 6, 7, 8, 11,
                9, 10, 14, 12, 13)
dist_mat = as.matrix(as.dist(ape::cophenetic.phylo(rt)))
dist_mat = dist_mat[index_order, index_order]
ade4::is.euclid(sqrt(as.dist(ape::cophenetic.phylo(rt))))
# standard double centering
w_taxa = rep(1 / nrow(dist_mat), nrow(dist_mat))
P_w_taxa = diag(x=rep(1,nrow(dist_mat))) - 
  matrix(data = rep(1,nrow(dist_mat)), ncol = 1) %*% t(w_taxa)
S = -0.5 * P_w_taxa %*% dist_mat %*% t(P_w_taxa)
decomp = eigen(S, TRUE)

df_rt <- data.frame(x=rep(1:50, 3), val=c(decomp$vectors[,1], 
                                          decomp$vectors[,2], 
                                          decomp$vectors[,3]), 
                    variable=rep(paste0("eigenvector", 1:3), each=50),
                    tree_type=rep("real tree", 150))

df = rbind(df_bt, df_ct)
df = rbind(df, df_rt)
ggplot2::ggplot(data = df, aes(x=x, y=val)) + geom_point() +
  facet_wrap(~tree_type+variable, scale="free") + cowplot::theme_half_open() + 
  cowplot::background_grid() + theme(legend.position = "none") +
  xlab("leaf node index") + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))
############################# end plots ########################################
