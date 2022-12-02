import os

def create_directory(dir_name):
	if str(type(dir_name)) != "<class 'str'>":
		print ('Error, input is not a string. Datatype: ', type(string))
	else:
		parent_directory = str(os.getcwd())
		directory = dir_name
		filepath = os.path.join(parent_directory, directory)
		if os.path.exists(filepath) != True:
			os.mkdir(filepath)
		return filepath + '/'

def create_subdirectory(parent, dir_name, sub_dir_name=''):
	if str(type(dir_name)) != "<class 'str'>":
		dir_name = str(dir_name)
	else:
		filepath = os.path.join(parent, dir_name)
		if os.path.exists(filepath) != True:
			os.mkdir(filepath)
		if sub_dir_name != '':
			filepath = os.path.join(filepath, sub_dir_name)
			if os.path.exists(filepath) != True:
				os.mkdir(filepath)
		return filepath + '/'

def name_checker(filepath, filename, file_ext=''):
	if str(type(file_ext)) != "<class 'str'>":
		file_ext = str(file_ext)
	name = untitled(filepath, filename, file_ext)
	check_name = name + file_ext
	path = os.path.join(filepath, check_name)
	if os.path.exists(path) == True:
		i = 2
		user_in = 'null'
		while True:
			user_in = input('A file of this name already exists! Do you want to rewrite it? [y/n]: ')
			if user_in == 'y':
				break
			elif user_in == 'n':
				break
			else:
				print('Please enter a valid input')
		if user_in == 'y':
			return name
		elif user_in == 'n':
			while os.path.exists(path) == True:
				new_name = name + ' ' + str(i)
				check_name = new_name + file_ext
				path = os.path.join(filepath, check_name)
				i = i + 1
			return new_name
	else:
		return name

def untitled(filepath, filename, file_ext):
	if filename == '':
		name = 'untitled'
		check_name = name + file_ext
		path = os.path.join(filepath, check_name)
		i = 2
		while os.path.exists(path) == True:
			name = 'untitled ' + str(i)
			check_name = name + file_ext
			path = os.path.join(filepath, check_name)
			i = i + 1
		return name
	else:
		return filename



if __name__ == '__main__':
	create_directory('test dir')
