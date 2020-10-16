"""
bee state space generator. Given population_size, write down all the states as a list s.t
start with ones and then follows from 0 to population_size - 1 list of states
"""


def gen_state_encoding(population_size):
    """ @author = Huy """
    state_encoding = [0, ]
    for i in range(1, population_size):
        state_encoding.extend([i, -i])
    return state_encoding


def map_state_code(state, state_encoding):
    """ @author = Huy """
    return [state_encoding[item] for item in state]


def backtrack(i, population_size, state, state_encoding) -> list:
    """ @author = Huy """
    results = []
    for v in range(len(state_encoding)):
        state[i] = v
        if i >= population_size - 1:
            results.append(map_state_code(state, state_encoding))
        else:
            sub_results = backtrack(i + 1, population_size, state, state_encoding)
            results.extend(sub_results)
    return results


def gen_full_statespace(population_size):
    """ @author = Huy """
    start_state = [0] * population_size
    results = backtrack(0, population_size, start_state, gen_state_encoding(population_size))
    return results


k_init_state_encoding = 3


def gen_semisync_statespace(population_size):
    """ @author = Huy """
    state_space = [tuple([k_init_state_encoding] * population_size), [1] * population_size]
    for i in range(population_size):
        init_state = [0] * (population_size - i)
        state_encoding = range(-i, 1)
        sub_state_space = backtrack(0, population_size - i, init_state, state_encoding)
        set_sub_state_space = set(tuple(sorted(state)) for state in sub_state_space)
        for s in set_sub_state_space:
            state = [1] * i + list(s)
            state_space.append(state)
    state_space = set(tuple(sorted(state, reverse=True)) for state in state_space)

    return state_space


def gen_async_statespace(population_size):
    """ @author = Huy """
    state_space = [tuple([k_init_state_encoding] * population_size), tuple([1] * population_size)]
    for i in range(population_size):
        init_state = [0] * (population_size - i)
        state_encoding = list(range(-i, 1))
        state_encoding.append(k_init_state_encoding)
        sub_state_space = backtrack(0, population_size - i, init_state, state_encoding)
        set_sub_state_space = set(tuple(sorted(state)) for state in sub_state_space)
        for s in set_sub_state_space:
            state = [1] * i + list(s)
            state_space.append(state)
    state_space = set(tuple(sorted(state, reverse=True)) for state in state_space)
    state_space = sorted(list(state_space))
    return state_space


if __name__ == "__main__":
    ss = gen_semisync_statespace(6)
    for s in ss:
        print(s)
    print(len(ss))
