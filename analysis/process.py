import pandas as pd
import numpy as np
import os

# # Define file paths
# file1 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset1_res/pdot1_dataset1_log_10000.csv'
# file2 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset1_res/pdot1_dataset1_log_3000.csv'
# file3 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset1_res/pdot1_dataset1_log_1003001000.csv'

# # Read the files and select columns
# df1 = pd.read_csv(file1)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df2 = pd.read_csv(file2)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df3 = pd.read_csv(file3)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]

# # Save the individual parts
# df1.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset1_part3.csv', index=False)
# df2.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset1_part2.csv', index=False)
# df3.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset1_part1.csv', index=False)

# # Concatenate the dataframes
# combined_df = pd.concat([df3, df2, df1])

# # Save the combined file
# combined_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset1.csv', index=False)






# # Define file paths
# file1 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset2_res/pdot1_dataset2_log_101732.csv'
# file2 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset2_res/pdot1_dataset2_log_55.csv'
# file3 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset2_res/pdot1_dataset2_log_100.csv'

# # Read the files and select columns
# df1 = pd.read_csv(file1)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df2 = pd.read_csv(file2)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df3 = pd.read_csv(file3)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]

# # Save the individual parts
# df1.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset2_part1.csv', index=False)
# df2.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset2_part2.csv', index=False)
# df3.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset2_part3.csv', index=False)

# # Concatenate the dataframes in the order part1, part2, part3
# combined_df = pd.concat([df1, df2, df3])

# # Save the combined file
# combined_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset2.csv', index=False)





# # Define file paths
# file1 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset1_res/pdot2_dataset1_log_100to1000.csv'
# file2 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset1_res/pdot2_dataset1_log_3000.csv'
# file3 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset1_res/pdot2_dataset1_log_10000.csv'

# # Read the files and select columns
# df1 = pd.read_csv(file1)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df2 = pd.read_csv(file2)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df3 = pd.read_csv(file3)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]

# # Save the individual parts
# df1.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset1_part1.csv', index=False)
# df2.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset1_part2.csv', index=False)
# df3.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset1_part3.csv', index=False)

# # Concatenate the dataframes
# combined_df = pd.concat([df1, df2, df3])

# # Save the combined file
# combined_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset1.csv', index=False)







# # Define file paths
# file1 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset2_res/pdot2_dataset2_log_10to32.csv'
# file2 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset2_res/pdot2_dataset2_log_55.csv'
# file3 = '/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset2_res/pdot2_dataset2_log_100.csv'

# # Read the files and select columns
# df1 = pd.read_csv(file1)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df2 = pd.read_csv(file2)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]
# df3 = pd.read_csv(file3)[['m', 'n', 'instance_number', 'primal_infeasibility', 'time_full']]

# # Save the individual parts
# df1.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2_part1.csv', index=False)
# df2.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2_part2.csv', index=False)
# df3.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2_part3.csv', index=False)

# # Concatenate the dataframes
# combined_df = pd.concat([df1, df2, df3])

# # Save the combined file
# combined_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2.csv', index=False)





# def read_matrix(file_path):
#     # Read the CSV file, using the first row as the header
#     return pd.read_csv(file_path, header=0, index_col=False).values


# def calculate_gap(answer_matrix, solution_matrix):
#     return np.abs(answer_matrix - solution_matrix).sum()

# # Define sizes and instances
# sizes = [100, 300, 1000, 3000, 10000]
# instances = range(1, 11)

# # List to store DataFrame rows
# rows = []

# for m in sizes:
#     for instance in instances:
#         solution_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset1_res/transport_matrix_instance{instance}_{m}x{m}.csv'
#         answer_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset1_answer/final_transport_matrix_instance{instance}_{m}x{m}.csv'

#         if os.path.exists(solution_file) and os.path.exists(answer_file):
#             solution_matrix = read_matrix(solution_file)
#             answer_matrix = read_matrix(answer_file)
#             gap = calculate_gap(answer_matrix, solution_matrix)
#             rows.append({'m': m, 'n': m, 'instance_number': instance, 'gap': gap})

# # Create DataFrame from rows
# results_df = pd.DataFrame(rows)

# # Save the results
# results_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset1_part4.csv', index=False)











# # Define the function to read the matrix from a file
# def read_matrix(file_path):
#     return pd.read_csv(file_path, header=0, index_col=False).values

# # Define the function to calculate the gap
# def calculate_gap(answer_matrix, solution_matrix):
#     return np.abs(answer_matrix - solution_matrix).sum()

# # Define sizes and instances
# sizes = [10, 17, 32, 55, 100]
# instances = range(1, 11)

# # List to store DataFrame rows
# rows = []

# for m in sizes:
#     for instance in instances:
#         solution_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot1_dataset2_res/transport_matrix_instance{instance}_{m}x{m}.csv'
#         answer_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset2_answer/final_transport_matrix_instance{instance}_{m}x{m}.csv'

#         if os.path.exists(solution_file) and os.path.exists(answer_file):
#             solution_matrix = read_matrix(solution_file)
#             answer_matrix = read_matrix(answer_file)
#             gap = calculate_gap(answer_matrix, solution_matrix)
#             rows.append({'m': m, 'n': m, 'instance_number': instance, 'gap': gap})

# # Create DataFrame from rows
# results_df = pd.DataFrame(rows)

# # Save the results
# results_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot1_dataset2_part4.csv', index=False)









# def read_matrix(file_path):
#     return pd.read_csv(file_path, header=0, index_col=False).values

# def calculate_gap(answer_matrix, solution_matrix):
#     return np.abs(answer_matrix - solution_matrix).sum()

# sizes = [100, 300, 1000, 3000, 10000]
# instances = range(1, 11)

# rows = []

# for m in sizes:
#     for instance in instances:
#         solution_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset1_res/transport_matrix_instance{instance}_{m}x{m}.csv'
#         answer_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset1_answer/final_transport_matrix_instance{instance}_{m}x{m}.csv'

#         if os.path.exists(solution_file) and os.path.exists(answer_file):
#             solution_matrix = read_matrix(solution_file)
#             answer_matrix = read_matrix(answer_file)
#             gap = calculate_gap(answer_matrix, solution_matrix)
#             rows.append({'m': m, 'n': m, 'instance_number': instance, 'gap': gap})

# results_df = pd.DataFrame(rows)
# results_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset1_part4.csv', index=False)










# def read_matrix(file_path):
#     return pd.read_csv(file_path, header=0, index_col=False).values

# def calculate_gap(answer_matrix, solution_matrix):
#     return np.abs(answer_matrix - solution_matrix).sum() 

# sizes = [10, 17, 32, 55, 100]
# instances = range(1, 11)

# rows = []

# for m in sizes:
#     for instance in instances:
#         solution_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/result/pdot2_dataset2_res/transport_matrix_instance{instance}_{m}x{m}.csv'
#         answer_file = f'/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset2_answer/final_transport_matrix_instance{instance}_{m}x{m}.csv'

#         if os.path.exists(solution_file) and os.path.exists(answer_file):
#             solution_matrix = read_matrix(solution_file)
#             answer_matrix = read_matrix(answer_file)
#             print("m:", m, "instance:", instance)
#             print("Solution matrix shape:", solution_matrix.shape)
#             print("Answer matrix shape:", answer_matrix.shape)  
#             gap = calculate_gap(answer_matrix, solution_matrix)
#             rows.append({'m': m, 'n': m, 'instance_number': instance, 'gap': gap})

# results_df = pd.DataFrame(rows)
# results_df.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2_part4.csv', index=False)


dataset1 = pd.read_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2.csv')
part4 = pd.read_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2_part4.csv')

# Check if the number of rows matches
if len(dataset1) == len(part4):
    # Add the 'gap' column from part4 to dataset1 
    dataset1['gap'] = part4['gap']

    # Save the updated dataset1
    dataset1.to_csv('/Users/justinmeng/Desktop/Project OR2/experiments/analysis/pdot2_dataset2.csv', index=False)
else:
    print("The number of rows in the datasets do not match.")






