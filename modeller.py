"""This file is for the purpose of running and testing different 
hyperperameters for the same model. It automatically logs the results 
so can be left running independently for long periods of time. """

import trainer

def model(lr, bs, gm):
    a1 = trainer.main(lr, bs, gm)
    print(a1)
    document(lr, bs, gm,a1)
         
def document(lr, bs, gm, accuracy):
    f = open('modelmodelling.txt','a')
    f.write('LR: '+ str(lr)+'\n')
    f.write('BS: '+ str(bs)+'\n')
    f.write('GM: '+ str(gm)+'\n')
    f.write('\n'+str(accuracy)+'\n')
    f.close()

model(2.5, 8, 0.9)
model(1.0, 8, 0.8)
model(2.5, 8, 0.9)
model(2.5, 8, 0.8)
