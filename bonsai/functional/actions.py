import polars as pl

from bonsai.functional.conditions import add_column, match


def action_router(df: pl.DataFrame, conditional_criteria: list, refs=None):
    refs = {} if refs is None else refs
    for cond in conditional_criteria:
        # Mode adjusts the input df
        if cond.mode == "match":
            output = match(df, **cond.args)
        elif cond.mode == "reference":
            output = refs[cond.args.name]
        elif cond.mode == "first":
            output = df.group_by("subject_id").agg(pl.col(cond.args.col).min())
        elif cond.mode == "last":
            output = df.group_by("subject_id").agg(pl.col(cond.args.col).max())
        elif cond.mode == "rename":
            output = df.rename(dict(cond.args))
        elif cond.mode == "shift_date":
            output = df.with_columns(
                pl.col(cond.args.col).dt.offset_by(cond.args.shift)
            )
        elif cond.mode == "add_column":
            output = add_column(df, **cond.args)
        # TODO: custom
        # elif cond.mode == "custom":
        #     output = custom_functions[cond.fn](df, **cond.args)
        else:
            raise ValueError(f"Mode={cond.mode} not supported")

        # Actions handle the output from `mode`
        if cond.action == "exclude":
            df = df.join(output, on="subject_id", how="anti")
        elif cond.action == "include":
            df = df.join(output, on="subject_id", how="semi")
        elif cond.action == "append":
            df = df.join(output, on="subject_id", how="left", validate="m:1")
        elif cond.action == "overwrite":
            df = output
        else:
            raise ValueError(f"Action={cond.action} not supported")
    return df
