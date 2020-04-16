def reward(owned_squares, old_targets, new_targets, id):

    rewards = []

    for i in range(len(owned_squares)):

        s = owned_squares[i]
        o = old_targets[i]
        n = new_targets[i]

        if(n.owner != id):
            rewards.append(-s.strength)

        elif(o.owner == 0 and n.owner == id):
            rewards.append((n.production + s.strength) * 10)

        else:
            rewards.append(-0.001)


    return rewards
