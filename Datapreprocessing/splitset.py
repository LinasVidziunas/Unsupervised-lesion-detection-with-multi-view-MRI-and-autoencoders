from random import shuffle


def split_set_by_patients(patients, train_amount: int, validation_amount: int,
                          test_amount: int):

    if len(patients) != train_amount + validation_amount + test_amount:
        raise ValueError(
            "Length of patients is not equal to the total amount of set amounts"
        )

    shuffle(patients)

    train_set = patients[0:train_amount]
    validation_set = patients[train_amount:train_amount + validation_amount]
    test_set = patients[train_amount + validation_amount:train_amount +
                        validation_amount + test_amount]

    return {"train": train_set, "validation": validation_set, "test": test_set}
