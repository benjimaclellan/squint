
#%%
# 
import ast

#%%
tree = ast.parse("print('hello world')")

tree = ast.Mod([ast.Expr()])

# %%
print(tree.body[0].__dict__)

#%%
obj = compile(tree, filename="<ast>", mode="exec")
exec(obj)