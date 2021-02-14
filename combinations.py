# Written by: Nick Gerend, @dataoutsider
# Viz: "Winter Forest", enjoy!

# Calculate slider combinations:
start = 0
end = 0
total = 127 # for 127 there are 8128 combinations
combinations = 0
for i in range(total):
    for j in range(start, total):
        # print('start: ' + str(start))
        # print('end: ' + str(end))
        end += 1
        combinations += 1
    start += 1
    end = start
print(combinations)