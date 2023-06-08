import synapseclient
syn = synapseclient.Synapse()
syn.login(email="zhangsuiyu657@gmail.com",password="Sy85009282")
dl_list_file_entities = syn.get_download_list()
