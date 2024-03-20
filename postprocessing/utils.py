

def format_code_path(code_add):
    '''
    This function complements the dir of csv files. Otherwise, the code will not be runnable since csv files like "atp_tennis.csv" is not under code dir.
    '''
    if "tapilot_data/atp_tennis.csv" not in code_add:
        code_add = code_add.replace('"atp_tennis.csv"', "os.path.join(sys.argv[1], 'atp_tennis.csv')")
        code_add = code_add.replace("'atp_tennis.csv'", "os.path.join(sys.argv[1], 'atp_tennis.csv')")
    if "tapilot_data/credit_customers.csv" not in code_add:
        code_add = code_add.replace("'credit_customers.csv'", "os.path.join(sys.argv[1], 'credit_customers.csv')")
        code_add = code_add.replace('"credit_customers.csv"', "os.path.join(sys.argv[1], 'credit_customers.csv')")
    if "tapilot_data/fastfood.csv" not in code_add:
        code_add = code_add.replace("'fastfood.csv'", "os.path.join(sys.argv[1], 'fastfood.csv')")
        code_add = code_add.replace('"fastfood.csv"', "os.path.join(sys.argv[1], 'fastfood.csv')")
    if "tapilot_data/laptops_price.csv" not in code_add:
        code_add = code_add.replace("'laptops_price.csv'", "os.path.join(sys.argv[1], 'laptops_price.csv')")
        code_add = code_add.replace('"laptops_price.csv"', "os.path.join(sys.argv[1], 'laptops_price.csv')")
    if "tapilot_data/melb_data.csv" not in code_add:
        code_add = code_add.replace("'melb_data.csv'", "os.path.join(sys.argv[1], 'melb_data.csv')")
        code_add = code_add.replace('"melb_data.csv"', "os.path.join(sys.argv[1], 'melb_data.csv')")

    return code_add


def format_code_gen_AIR(code_add, change_flg):
    change_flg = True
    for line in code_add.split("\n"):
        if "def read_csv_file(" in line:
            change_flg = False
            continue
        if "read_csv_file(" in line or "pd.read_csv(" in line:
            if change_flg:
                code_add = code_add.replace(line, "")
            else:
                change_flg = True

    return code_add, change_flg