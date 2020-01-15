E_PATH='/home/lorenzo/Documenti/UPF/DeepRL/Protocols/Test_transfer_learning_clean_nodes_KeyDoor_change/'
prefix='/home/lorenzo/Documenti/UPF/DeepRL/Protocols/'

for i in $(find $E_PATH -name '*.py' -type f)
do 
	PROT_PATH=${i#"$prefix"}
	PROT_PATH="$(tr / . <<<$PROT_PATH)"
	echo $PROT_PATH
done



