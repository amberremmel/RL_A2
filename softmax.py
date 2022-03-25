elif policy == 'softmax':
if temp is None:
    raise KeyError("Provide a temperature")
# TO DO: Add own code
x = self.Q_sa[s]
y = softmax(x, temp)
z = random.random()
for i, val in enumerate(y):
    if z < val:
        a = i
        break
    else:
        z -= val
# a = np.random.randint(0,self.n_actions) # Replace this with correct action selection

return a