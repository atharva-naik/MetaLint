# byr (Birth Year)
# iyr (Issue Year)
# eyr (Expiration Year)
# hgt (Height)
# hcl (Hair Color)
# ecl (Eye Color)
# pid (Passport ID)
# cid (Country ID)

required_keys = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'}
optional_keys = {'cid'}
filename = "./d4/input.txt"
passports = []

with open(filename) as f:
    passports = []
    passport = {}
    for line in f.readlines():
        if len(line.strip()) == 0:
            passports.append(passport)
            passport = {}
        else:
            fields = line.strip().split(' ')
            for field in fields:
                f, v = field.split(':')
                passport.update({f:v})
       
    if passports:
        passports.append(passport)

# byr (Birth Year) - four digits; at least 1920 and at most 2002.
# iyr (Issue Year) - four digits; at least 2010 and at most 2020.
# eyr (Expiration Year) - four digits; at least 2020 and at most 2030.
# hgt (Height) - a number followed by either cm or in:
# If cm, the number must be at least 150 and at most 193.
# If in, the number must be at least 59 and at most 76.
# hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
# ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
# pid (Passport ID) - a nine-digit number, including leading zeroes.
# cid (Country ID) - ignored, missing or not.

def validate_passport(passport):

    def _check_valid_int(value, count, lb, ub):
        if len(value) != count:
            return False
        ivalue = int(value)
        if ivalue < lb or ivalue > ub:
            return False
        return True
    byr = passport['byr']
    if not _check_valid_int(byr, 4, 1920, 2002):
        return False

    iyr = passport['iyr']
    if not _check_valid_int(iyr, 4, 2010, 2020):
        return False

    eyr = passport['eyr']
    if not _check_valid_int(eyr, 4, 2020, 2030):
        return False
    

def count_valid(passports, required_keys, optional_keys):

    count = 0
    for p in passports:
        s = set(p.keys())
        if len(required_keys - optional_keys - s) == 0:
            count += 1
    return count

print(f'valid count: {count_valid(passports, required_keys, optional_keys)}')