# decode_message(file='coding_qual_input.txt')


"""
- iterate over the list and extract values from the list based on the step 

"""


def decode_message(file) -> None:
    def create_staircase(nums):
        step = 1
        subsets = []
        while len(nums) != 0:
            if len(nums) >= step:
                subsets.append(nums[0:step])
                nums = nums[step:]
                step += 1
            else:
                return False
        return subsets

    code = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            values = line.split()
            code.update({f"{values[0]}": f"{values[1]}"})
    sorted_list = sorted([int(i) for i in code.keys()])
    list_code = [str(i[-1]) for i in create_staircase(sorted_list)]
    message = [code.get(f"{i}") for i in list_code]
    message = ' '.join(message)
    print(message)
    return message






decode_message(file="coding_qual_input.txt")