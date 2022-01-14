######################### load required packages ###############################

library(ade4)
library(MatrixCorrelation)
library(Matrix)
library(ggplot2)
library(ape)
library(ggtree)
library(treeio)
library(tidyr)
library(tibble)
library(forcats)
library(ggpubr)
library(phyloseq)
library(adaptiveGPCA)
library(vegan)
library(treeDA)

######################### end load packages ####################################

# read the comb tree
ct <- treeio::read.beast.newick("../data/ctree.tree")

ct_plot <- ggtree(ct) + layout_dendrogram()

c_largest <- read.csv("../data/comb_largest_10000.csv", header = TRUE)
c_second <- read.csv("../data/comb_second_10000.csv", header = TRUE)
c_third <- read.csv("../data/comb_third_10000.csv", header = TRUE)

# toggling the eigenvectors for better visulization
c_second$k2 <- -c_second$k2
c_second$k5 <- -c_second$k5
c_second$k6 <- -c_second$k6
c_second$k7 <- -c_second$k7
c_third$k1 <- -c_third$k1
c_third$k2 <- -c_third$k2
c_third$k4 <- -c_third$k4

colnames(c_largest) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(c_second) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(c_third) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')

c_largest = c_largest %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_largest$index <- as.numeric(c_largest$index)
c_largest$eig = rep('eigenvector 1', 224)
c_second = c_second %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_second$index <- as.numeric(c_second$index)
c_second$eig = rep('eigenvector 2', 224)

c_third = c_third %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_third$index <- as.numeric(c_third$index)
c_third$eig = rep('eigenvector 3', 224)

ct_df <- rbind(c_largest, c_second, c_third)

# comb tree plot and eigenvectors plot
p2 <- ggplot(subset(ct_df, eig %in% c('eigenvector 1')), aes(index, values)) + 
  geom_point() + 
  labs(x = "", y = "") + 
  facet_grid(k~eig) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))
p3 <- ggplot(subset(ct_df, eig %in% c('eigenvector 2')), aes(index, values)) + 
  geom_point() + 
  labs(x="", y = "") + 
  facet_grid(k~eig) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))
p4 <- ggplot(subset(ct_df, eig %in% c('eigenvector 3')), aes(index, values)) + 
  geom_point() + 
  labs(x="", y = "") + 
  facet_grid(k~eig) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))

ct_one_first <- treeDA::combine_plot_and_tree(p2, ct_plot)
ct_one_second <- treeDA::combine_plot_and_tree(p3, ct_plot)
ct_one_third <- treeDA::combine_plot_and_tree(p4, ct_plot)

ct_better <- ggarrange(ct_one_first, ct_one_second, ct_one_third,
                       ncol = 3, nrow = 1) 
annotate_figure(ct_better, bottom = text_grob("leaf node index"))

# read the real tree

load("../data/phyloseq_objects.RData")
set.seed(0)
indx = sample(c(1:1651), 50)
indx = sort(indx)
logic_vec = c(1:1651)

pos = 1
for (i in logic_vec){
  if (i == indx[pos] & pos <= 50){
    pos = 1 + pos
    logic_vec[i] = TRUE
  } else{
    logic_vec[i] = FALSE
  }
}
logic_vec = as.logical(logic_vec)

small = prune_taxa(logic_vec, AntibioticPhyloseq)
small_tree = phy_tree(small)
rt_plot <- ggtree(small) + layout_dendrogram() + geom_tiplab(size =1.5)

r_largest <- read.csv("../data/real_largest.csv", header = TRUE)
r_second <- read.csv("../data/real_second.csv", header = TRUE)
r_third <- read.csv("../data/real_third.csv", header = TRUE)

# reordering the tip index to match two plots
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
r_largest = r_largest[index_order,]
r_second = r_second[index_order,]
r_third = r_third[index_order,]
colnames(r_largest) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(r_second) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(r_third) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
r_largest = r_largest %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)

r_second = r_second %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)

r_third = r_third %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)

r_largest$eig = rep('eigenvector 1', 350)
r_second$eig = rep('eigenvector 2', 350)
r_third$eig = rep('eigenvector 3', 350)
rt_df <- rbind(r_largest, r_second, r_third)

p2 <- ggplot(subset(rt_df, eig %in% c('eigenvector 1')), aes(index, values)) + 
  geom_point() + 
  aes(x = fct_inorder(index)) + 
  labs(x="", y = "") + 
  facet_grid(k~eig)+ theme_bw() +
  theme(axis.text.x=element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))

p3 <- ggplot(subset(rt_df, eig %in% c('eigenvector 2')), aes(index, values)) + 
  geom_point() + 
  aes(x = fct_inorder(index)) + 
  labs(x="", y = "") + 
  facet_grid(k~eig)+ theme_bw() +
  theme(axis.text.x=element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))

p4 <- ggplot(subset(rt_df, eig %in% c('eigenvector 3')), aes(index, values)) + 
  geom_point() + 
  aes(x = fct_inorder(index)) + 
  labs(x="", y = "") + 
  facet_grid(k~eig)+ theme_bw() +
  theme(axis.text.x=element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))

rt_one_first <- treeDA::combine_plot_and_tree(p2, rt_plot)
rt_one_second <- treeDA::combine_plot_and_tree(p3, rt_plot)
rt_one_third <- treeDA::combine_plot_and_tree(p4, rt_plot)
rt_better <- ggarrange(rt_one_first, rt_one_second, rt_one_third,
                       ncol = 3, nrow = 1) 
annotate_figure(rt_better, bottom = text_grob("leaf node index"))

# eigenvalues plot

b_eigvals <- read.csv("../data/b_eigvals_10000.csv", header = TRUE)
c_eigvals <- read.csv("../data/c_eigvals_10000.csv", header = TRUE)
r_eigvals <- read.csv("../data/r_eigvals.csv", header = TRUE)
b_eigvals  = as.data.frame(lapply(b_eigvals, function(x) x/sum(x)))
c_eigvals  = as.data.frame(lapply(c_eigvals, function(x) x/sum(x))) 
r_eigvals  = as.data.frame(lapply(r_eigvals, function(x) x/sum(x))) 
colnames(b_eigvals) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(c_eigvals) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
colnames(r_eigvals) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
b_eigvals = b_eigvals %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_eigvals = c_eigvals %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
r_eigvals = r_eigvals %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
b_eigvals$tree = rep('binary', 224)
c_eigvals$tree = rep('comb', 224)
r_eigvals$tree = rep('real', 350)
b_eigvals$index <- as.numeric(b_eigvals$index)
c_eigvals$index <- as.numeric(c_eigvals$index)
r_eigvals$index <- as.numeric(r_eigvals$index)

bp1 <- ggplot(b_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black")) +
  xlab("order") + ylab("ratio") + ggtitle("binary tree")

bp2 <- ggplot(b_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + 
  coord_cartesian(xlim = c(2, 9), ylim = c(0, 0.15)) + theme_bw() +
  theme(legend.position = "none", panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black")) + xlab("") + ylab("")


bp_zoom = ggplotGrob(bp2)
bp = bp1 + annotation_custom(grob = bp_zoom, xmin = 7, 
                             xmax = 32, ymin = 0.2, ymax = 0.7)

cp1 <- ggplot(c_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))+
  xlab("order") + ylab("ratio") + ggtitle("comb tree")

cp2 <- ggplot(c_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + 
  coord_cartesian(xlim = c(2, 9), ylim = c(0,0.08)) + theme_bw() +
  theme(legend.position = "none", panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black")) + xlab("") + ylab("")


cp_zoom = ggplotGrob(cp2)
cp = cp1 + annotation_custom(grob = cp_zoom, xmin = 7, 
                             xmax = 32, ymin = 0.2, ymax = 0.7)

rp1 <- ggplot(r_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"))+
  xlab("order") + ylab("ratio") + ggtitle("real tree")

rp2 <- ggplot(r_eigvals, 
              aes(index, values, group = k)) + geom_path(aes(color = k)) +
  geom_point(aes(color = k)) + 
  coord_cartesian(xlim = c(2, 9), ylim = c(0, 0.1)) + theme_bw() +
  theme(legend.position = "none", panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black")) + xlab("") + ylab("")


rp_zoom = ggplotGrob(rp2)
rp = rp1 + annotation_custom(grob = rp_zoom, xmin = 7, xmax = 48, 
                             ymin = 0.23, ymax = 0.83)


p_eig <- ggarrange(bp, cp, rp,
                  ncol = 3, nrow = 1, common.legend = TRUE, legend = 'bottom')
p_eig

# eigenvectors plots in the supplementary for larger trees

# binary tree

b_1 <- read.csv("../data/binary_largest_128tips_seqlen7.csv", header = TRUE)
colnames(b_1) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
b_1 = b_1 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
b_1$eigenvector = rep('eigenvector 2', 128 * 7)
b_1$index <- as.numeric(b_1$index)

b_2 <- read.csv("../data/binary_second_128tips_seqlen7.csv", header = TRUE)
colnames(b_2) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
b_2 = b_2 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
b_2$eigenvector = rep('eigenvector 3', 128 * 7)
b_2$index <- as.numeric(b_2$index)

b_3 <- read.csv("../data/binary_third_128tips_seqlen7.csv", header = TRUE)
colnames(b_3) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
b_3 = b_3 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
b_3$eigenvector = rep('eigenvector 4', 128 * 7)
b_3$index <- as.numeric(b_3$index)

b_df <- rbind(b_1, b_2, b_3)

ggplot(data = b_df, aes(index, values)) + 
  geom_point(aes(color = eigenvector)) + facet_wrap(~k) + theme_bw() +
  theme(legend.position = "bottom") +
  xlab("Tip Index")

# comb tree

c_1 <- read.csv("../data/comb_largest_128tips_seqlen7.csv", header = TRUE)
colnames(c_1) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
c_1 = c_1 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_1$eigenvector = rep('eigenvector 1', 128 * 7)
c_1$index <- as.numeric(c_1$index)

c_2 <- read.csv("../data/comb_second_128tips_seqlen7.csv", header = TRUE)
c_2$k5 <- -c_2$k5
c_2$k6 <- -c_2$k6
colnames(c_2) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
c_2 = c_2 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_2$eigenvector = rep('eigenvector 2', 128 * 7)
c_2$index <- as.numeric(c_2$index)

c_3 <- read.csv("../data/comb_third_128tips_seqlen7.csv", header = TRUE)
c_3$k2 <- -c_3$k2
c_3$k3 <- -c_3$k3
c_3$k6 <- -c_3$k6
colnames(c_3) <- c('k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7')
c_3 = c_3 %>% 
  rownames_to_column("index") %>%
  gather(k, values, -index)
c_3$eigenvector = rep('eigenvector 3', 128 * 7)
c_3$index <- as.numeric(c_3$index)

c_df <- rbind(c_1, c_2, c_3)

ggplot(data = c_df, aes(index, values)) + 
  geom_point(aes(color = eigenvector)) + facet_wrap(~k) + theme_bw() +
  theme(legend.position = "bottom") +
  xlab("Tip Index")

