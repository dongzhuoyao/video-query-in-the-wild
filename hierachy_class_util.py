cls_dict = dict()

father_set = set()
gfather_set = set()
ggfather_set = set()
with open("activity_net_depth_v5.csv") as f:
    lis = [line.split(",") for line in f]        # create a list of lists
    for i, x in enumerate(lis):              #print the list items
        print("line{0} = {1}".format(i, x))
        cls_dict[x[0]] = dict(
            father=x[1],
            grandfather=x[2],
            grandgrandfather =x[3]
        )
        father_set.add(x[1])
        gfather_set.add(x[2])
        ggfather_set.add(x[3])


def is_same_father(a,b):
    if a == 'noisy_activity' or b == 'noisy_activity':
        return False
    return cls_dict[a]['father'] == cls_dict[b]['father']

def is_same_grandfather(a,b):
    if a == 'noisy_activity' or b == 'noisy_activity':
        return False
    return cls_dict[a]['grandfather'] == cls_dict[b]['grandfather']

def is_same_grandgrandfather(a,b):
    if a == 'noisy_activity' or b == 'noisy_activity':
        return False
    return cls_dict[a]['grandgrandfather'] == cls_dict[b]['grandgrandfather']

if __name__ == '__main__':
    print(is_same_grandgrandfather("Sumo",'Arm wrestling'))
    print(is_same_father('Waterskiing', 'Arm wrestling'))
    print(len(father_set))
    print(len(gfather_set))
    print(len(ggfather_set))



