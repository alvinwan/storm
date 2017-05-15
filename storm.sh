export CKPT_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)

if [ $1 = "encode_ae" ]; then
	cd src && \
	python neuralnetworks/autoencoder3d.py train ../data/train_$2.mat $CKPT_ID && \
	python neuralnetworks/autoencoder3d.py encode $CKPT_ID ../data/train_$2.mat ../data/generated/train_$2_encoded_ae.mat && \
	python neuralnetworks/autoencoder3d.py encode $CKPT_ID ../data/test_$2.mat ../data/generated/test_$2_encoded_ae.mat
elif [ $1 = "encode_kmeans" ]; then
	cd src && \
	python kmeans.py train ../data/train_$2.mat $CKPT_ID && \
	python kmeans.py encode $CKPT_ID ../data/train_$2.mat ../data/generated/train_$2_encoded_kmeans.mat && \
	python kmeans.py encode $CKPT_ID ../data/test_$2.mat ../data/generated/test_$2_encoded_kmeans.mat
elif [ $1 = "encode_pca" ]; then
    cd src && \
	python pca.py train ../data/train_$2.mat $CKPT_ID && \
	python pca.py encode $CKPT_ID ../data/train_$2.mat ../data/generated/train_$2_encoded_pca.mat && \
	python pca.py encode $CKPT_ID ../data/test_$2.mat ../data/generated/test_$2_encoded_pca.mat
elif [ $1 = "svm" ]; then
    cd src && \
	python svm.py train ../data/generated/train_$2_encoded_$3.mat ../data/generated/test_$2_encoded_$3.mat --grid
elif [ $1 = "classify" ]; then
    python svm.py classify $2
fi