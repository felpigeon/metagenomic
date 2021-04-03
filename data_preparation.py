import numpy as np
import pandas as pd

def uniq_id(name):
  dataset, subject, sample = [], [], []
  disease = []

  with open(name, 'r') as  f:
    for l in f.readlines():
      info = l[:-1].split('\t')
      attribut_name = info[0]

      if attribut_name == 'dataset_name':
        dataset = info[1:]
      elif attribut_name == 'sampleID':
        sample = info[1:]
      elif attribut_name == 'subjectID':
        subject = info[1:]
      elif attribut_name == 'disease':
        disease = info[1:]

        break   #disease est la dernière ligne qui nous intéresse ici

  ids = [ '{}/{}/{}'.format(d,sub,sam) for d,sub,sam in zip(dataset, subject, sample)]
  return ids, disease


# On utilise une fonction pour ne par garder 2 copies des
# id et des maladies en mémoire pour rien.
def load_and_validate_ids():
  abundance_ids, ab_disease = uniq_id('abundance.txt')
  marker_ids, ma_disease = uniq_id('marker_presence.txt')

  assert( all([m == a for a,m in zip(abundance_ids, marker_ids)])  )
  assert( all([m == a for a,m in zip(ab_disease, ma_disease)])  )

  return marker_ids, ma_disease 


def split_dataset(split_size = 0.8):
	ids, disease = load_and_validate_ids()

	#df est un standard pour DataFrame
	df = pd.DataFrame( [i.split('/') for i in ids],
                    columns=['dataset', 'subject', 'sample'] )

	#On retire les doublons (en considérant les colonnes dataset et subject)
	df = df.drop_duplicates(['dataset', 'subject'])

	train_idx, test_idx = [], []

	for dataset in set(df.dataset):
		points = df.loc[ df.dataset == dataset ].index.values
		np.random.shuffle( points )

		borne = int( len(points) * split_size )
		train_idx += points[:borne].tolist()
		test_idx += points[borne:].tolist()

	train_idx = np.array(train_idx)
	test_idx = np.array(test_idx)

	#On mélange les échantillons des études
	np.random.shuffle( train_idx )
	np.random.shuffle( test_idx )

	np.save('data/train_idx.npy', train_idx)
	np.save('data/test_idx.npy', test_idx)

	print('Training set size:', len(train_idx))
	print('Testing set size:', len(test_idx))




def get_n_features(file_name, prefix):
  n_features = 0

  with open(file_name, 'r') as  f:
    for l in f.readlines():
      if not l.startswith(prefix):continue
      n_features += 1

  return n_features


def load_all_features(name, prefix, ids, features_index):
	features_pointer = 0
	file_pointer = 0

	if features_index is None:
		n_features = get_n_features(name, prefix)
	else:
		n_features = len(features_index)

	features = np.zeros( (len(ids), n_features) )

	with open(name, 'r') as f :
		for line in f.readlines():
			if not line.startswith(prefix):
				continue

			if  features_index is not None and not (file_pointer in features_index):
				file_pointer+= 1
				continue

			info = line[:-1].split('\t')
			x_line = np.array([float(i) for i in info[1:]])
			features[ :, features_pointer ] = x_line[ids]

			features_pointer+= 1
			file_pointer+= 1

	return features



def load_x_and_y(index_path, features_path=None):
	index = np.load(index_path)

	if features_path:
		features = np.load(features_path)
		features_marker = features[features < 288347]
		features_abundance = features[features >= 288347] - 288347
	else:
		features_marker = None
		features_abundance = None

	x = load_all_features( 'marker_presence.txt', 'gi|', index, features_marker)
	print( 'marker features:', x.shape[1] )

	x1 = load_all_features( 'abundance.txt', 'k__', index, features_abundance)
	print( 'abundances features:', x1.shape[1] )

	x = np.concatenate( [x, x1], axis=-1 )
	print( 'total n of features:', x.shape[1] )

	_, disease = uniq_id('abundance.txt')

	maladies = ['t2d', 'ibd_crohn_disease', 'ibd_ulcerative_colitis',
			'stec2-positive', 'obesity', 'overweight', 'underweight', 'obese',
			'ibd_crohn_disease', 'ibd_ulcerative_colitis', 'impaired_glucose_tolerance', 'y',
			'large_adenoma', 'small_adenoma', 'cancer', 'cirrhosis']

	# Classe: 0=malade, 1=en santée
	y = np.array( [1-int(d in maladies) for d in disease] )
	y = y[index]

	return x, y


if __name__ == '__main__':
	split_dataset()
