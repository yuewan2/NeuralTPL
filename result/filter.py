

gen_tpls = set()
with open('generated_templates.txt', 'r') as f:
  i = 0
  while True:
    line = f.readline() 
    if line.rstrip() not in gen_tpls:
      gen_tpls.add(line.rstrip())
    i += 1
    if i > 453098774:
      break 
    if i % 20000 == 0:
      print(i)

with open('generated_templates_filtered.txt', 'w') as f:
  for tpl in gen_tpls:
    f.write(tpl)
    f.write('\n')
print(len(gen_tpls))    


