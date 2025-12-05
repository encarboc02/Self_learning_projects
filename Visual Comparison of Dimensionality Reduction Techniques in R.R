
##########
# Parkinson Dimensionality Reduction Analysis
## PCA, t-SNE, Isomap, UMAP
##########


library (dplyr)
library (ggplot2)
library(caret)
library (stats)
library (factoextra)
library (Rtsne)
library (RDRToolbox)
library(uwot)
library(gridExtra)

#--------------Dataset and Preprocessing------------

data <- read.csv("parkinsons.data", sep = ',')

labels <- factor(data$status)
data_num <- data%>% select(-name, -status)

preproce <- preProcess(data_num, method = c('center','scale'))
data_pro <- predict(preproce, newdata = data_num)


#----------------------PCA---------------------

pca_results <- prcomp(data_pro)
pca_data <- data.frame(pca_results$x)

##Variance
var <- pca_results$sdev^2
tot_var <- sum(var)
var_ex <- var/tot_var
var_ac <- cumsum(var_ex)

# First PCs covering ≥80% variance
n_pca <- min(which(var_ac > 0.8))
print(n_pca)

#PCs importance
fviz_eig(pca_results, addlabels = T, ylim= c(0,50), ncp = 15)


#Labels
x_label <- paste0(paste('PC1', round(var_ex[1]*100, 2)), '%')
y_label <- paste0(paste('PC2', round(var_ex[2]*100, 2)), '%')

pca_plot<- ggplot(pca_data, aes(x=PC1,y=PC2, color = labels))+
              geom_point(size=3)+
              labs(title = 'PCA Parkinson', x=x_label, y= y_label, color = labels)+
              theme_minimal()

fviz_contrib(pca_results, choice = "var", axes = 1:2)


#---------------------------t-SNE------------------------

set.seed(2002)

tsne <- Rtsne(X=data_pro, dims = 2, perplexity=35)
tsne_result <- data.frame(tsne$Y)

# Graficamos
tsne_plot<- ggplot(tsne_result, aes(x = X1, y = X2, color = labels)) +
              geom_point(size = 3) +
              labs(title = "t-SNE Parkinson", x = "X1", y = "X2", color = labels) +
              theme_minimal()


#-----------------------ISOMAP----------------------

data_mat <- as.matrix(data_pro)
iso_results <- Isomap(data = data_mat, dims = 2, k=15)

iso_df<- data.frame(iso_results$dim2)

iso_plot<- ggplot(iso_df, aes(x=X1, y=X2, color=labels))+
              geom_point(size=3)+
              labs(title = "ISOMAP Parkinson", x = "X1", y = "X2", color = labels) +
              theme_minimal()


#--------------------UMAP---------------------

set.seed(2002)

umap_results <- umap(data_pro, n_neighbors=15, n_components = 2, min_dist = 0.05, local_connectivity=1, ret_model = TRUE)
umap_result2s <- umap(data_pro, n_neighbors=15, n_components = 2, min_dist = 0.05, local_connectivity=1, ret_model = TRUE, metric = "manhattan")

umap_df <- data.frame(umap_results$embedding) #dataframe euclidean
umap_df2 <- data.frame(umap_result2s$embedding) #dataframe manhattan


umap_eu_plot<- ggplot(umap_df, aes(x = X1, y = X2, color = labels)) +
                  geom_point(size = 3) +
                  labs(title="UMAP_euclidean Parkinson", x= "Dim1", y= "Dim2") +
                  theme_minimal() 

umap_man_plot<- ggplot(umap_df2, aes(x = X1, y = X2, color = labels)) +
                  geom_point(size = 3) +
                  labs(title="UMAP_manhattan Parkinson", x= "Dim1", y= "Dim2") +
                  theme_minimal()


#-------------------------Comparative visualization (grid)----------------

grid.arrange(pca_plot, tsne_plot, iso_plot, umap_eu_plot, umap_man_plot, ncol=2)

all_plot<- arrangeGrob(pca_plot, tsne_plot, iso_plot, umap_eu_plot, umap_man_plot, ncol=2)
ggsave("Comparison_plot.png", all_plot, width=12, height = 10, dpi = 100)
