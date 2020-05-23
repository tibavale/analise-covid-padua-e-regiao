#!/usr/bin/python

from scipy.integrate import odeint
import numpy
import matplotlib.pyplot as plt
#%matplotlib inline 
#!pip install mpld3
import mpld3
import re
mpld3.enable_notebook()


####################################
def plotseird(t, S, E, I, R, D=None, L=None, R_0=None, Alpha=None):
  #f, ax = plt.subplots(1,1,figsize=(10,4))
  plt.figure(0,figsize=(10,4))
  plt.clf()
  plt.plot(t, S, 'b', alpha=0.7, linewidth=1.5, label='Suscetiveis')
  plt.plot(t, E, 'y', alpha=0.7, linewidth=1.5, label='Expostos')
  plt.plot(t, I, 'r', alpha=0.7, linewidth=1.5, label='Infectados')
  plt.plot(t, R, 'g', alpha=0.7, linewidth=1.5, label='Recuperados')
  if D is not None:
    plt.plot(t, D, 'k', alpha=0.7, linewidth=1.5, label='Mortos')
    plt.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=1.5, label='Total')
  else:
    plt.plot(t, S+E+I+R, 'c--', alpha=0.7, linewidth=1.5, label='Total')
  
  
  plt.xlabel('Tempo (dias)')
  #plt.xlabel('Contagem de ')
  

  plt.grid(b=True, which='major', c='w', lw=5, ls='-')
  legend = plt.legend(borderpad=2.0,loc='center left')
  legend.get_frame().set_alpha(0.5)
  m="SL-"
  plt.title("Modelo SEIRD aplicado a populacao de "+str(nome_cidade))
  if L is None:
    plt.plot(data,confirm,'ro',markersize=4,label='Casos Confirmados')
    plt.plot(data,obitos,'ko',markersize=4,label='Obitos Registrados')
    #esp=obitos.astype(float)/0.02
    #plt.plot(data,esp,'o',color='pink',markersize=4,label='Valor esperado de confirm')
    #ax.plot(data,recup,'go',markersize=4,label='Recuperados')
  else:
    m="CL-"
    plt.title("Modelo SEIRD aplicado a populacao de "+str(nome_cidade)+" (Quarentena apos {} dias)".format(L))

  plt.show(block=False)
  plt.savefig("modelo-"+str(m)+str(cidade)+".pdf",dpi=200)
  plt.savefig("modelo-"+str(m)+str(cidade)+".png",dpi=200)

  if L is None:
    plt.xlim(0,t[-1])
    plt.xlim(0,data[-1])
    plt.xlabel('Data')
    k=numpy.where(t<=data[-1])[0]
    plt.ylim(0,I[k[-1]])
    #plt.ylim(0,I.max()*1.1)
    #plt.ylim(0,esp.max()*1.5)
    plt.subplots_adjust(bottom=0.20)
    plt.xticks(data,texto.T[0][z],rotation='vertical')
    #legend = plt.legend(borderpad=2.0,loc='center right')
  else:
    #plt.xlim(0,t[-1])
    plt.ylim(0,R[-1])
    #plt.ylim(0,I.max()*1.1)
    #plt.ylim(0,esp.max()*1.5)
    
  legend.get_frame().set_alpha(0.5)
  plt.savefig("modelo-zoom-"+str(m)+str(cidade)+".pdf",dpi=200)
  plt.savefig("modelo-zoom-"+str(m)+str(cidade)+".png",dpi=200)
  

  #if R_0 is not None or CFR is not None:
  #  f = plt.figure(figsize=(12,4))
  
  if R_0 is not None:
    plt.figure(1,figsize=(10,4))
    plt.clf()
    plt.plot(t, R_0, 'b--', alpha=0.7, linewidth=1.5, label='R$_0$')

    plt.xlabel('Tempo (dias)')
    plt.ylabel('Individuos')
    plt.title('R_0 como funcao do tempo')
    plt.grid(b=True, which='major', c='w', lw=10, ls='-')
    legend = plt.legend()
    legend.get_frame().set_alpha(0.5)
    plt.title("$R_0$")
    plt.show(block=False);
    plt.savefig("R_0-"+str(cidade)+".pdf",dpi=200)

  if ((Alpha is not None) and (alpha==alpha_opt)):
    f, ax2 = plt.subplots(1,1,figsize=(6,4))
    ax2.plot(t, Alpha_over_time, 'r--', alpha=0.7, linewidth=1.5, label='alpha')

    ax2.set_xlabel('Tempo (dias)')
    ax2.title.set_text('Letalidade no tempo')
    ax2.yaxis.set_tick_params(length=4)
    ax2.xaxis.set_tick_params(length=4)
    ax2.grid(b=True, which='major', c='w', lw=10, ls='-')
    legend = ax2.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
      plt.spines[spine].set_visible(False)    
    #plt.title("$\\alpha$")
    #plt.show(block=False)
    #plt.savefig("alpha-"+str(cidade)+".pdf",dpi=200)
    
    
################################################
# Time-Dependent R0
# Advanced Approach: logistic R0



def deriv1(y, t, N, beta, gamma, delta, alpha, rho):
    S, E, I, R, D = y
    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt
####################################
def deriv2(y, t, N, beta, gamma, delta, alpha_opt, rho):
    S, E, I, R, D = y
    def alpha(t):
        return s * I/N + alpha_opt
    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha(t)) * gamma * I - alpha(t) * rho * I
    dRdt = (1 - alpha(t)) * gamma * I
    dDdt = alpha(t) * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt
####################################
def logistic_R_0(t):
    R_0=(R_0_start-R_0_end) / (1 + numpy.exp(-coeff*(-t+x0))) + R_0_end
    if L is not None:
      if(numpy.isscalar(t)):
        if( t >= L ):
            R_0=R_0_end
      else:
        R_0[numpy.where( t >= L )[0]]=R_0_end
    return R_0
####################################
def beta(t):
    return logistic_R_0(t) * gamma

####################################
def subnotificacao():
	
	#k=numpy.where(t<=data[-1])[0]
	#tt=numpy.array([int(round(i)) for i in t])
	#print tt
	#
	#j=numpy.array([numpy.where(data==tt[o])[0][0] for o in k])
	#h=numpy.array([])
	#temp=tt[k]
	#jj=numpy.array([numpy.where(temp==o)[0][0] for o in data[j]])
	#print data[j],"\n\n",confirm[j],"\n\n",I[jj]
	#subnotif=100*(I[jj]-confirm[j].astype(float))/I[jj]
	
	k=numpy.array([numpy.where(t==o)[0][0] for o in data])
	
	subnotif=100*(I[k]-confirm.astype(float))/I[k]
	
	plt.figure(2,figsize=(15,10))
	plt.clf()
	plt.plot(data,subnotif,'bo-',markersize=4)
	plt.ylim(subnotif.min()*0.9,100)
	plt.xticks(data,texto.T[0],rotation='vertical')
	vals=numpy.arange(0,101,10)
	plt.yticks(vals,['{:}%'.format(x) for x in vals])
	plt.grid()
	plt.subplots_adjust(bottom=0.20)
	plt.xlabel('Datas')
	plt.ylabel('Percentual estimado')
	plt.title('Estimativa de subnotificacao de COVID-19 em '+str(nome_cidade))
	plt.show(block=False)
	plt.savefig('subnotificacao-'+str(cidade)+'.pdf',dpi=200)

####################################

#Municipio, Populacao, E0, I0, R0, R_0_start, R_0_end

entrada=numpy.array([
#["Rio-de-Janeiro",6718903,500,200,2.8,0.9],
#["Wuhan",11000000,500,200,2.8,0.9],
["Campos-dos-Goytacazes",507548,8,4,2.8,0.9],
["Itaocara",23234,10,5,3.0,0.9],
["Itaperuna",103224,4,2,2.5,0.9],
["Miracema",27174,2,1,2.8,0.9],
["Pirapetinga",10752,1,1,1.9,0.9],
["Santo-Antonio-de-Padua",42479,2,1,3.5,0.9],
["Sao-Fidelis",38669,8,4,2.8,0.9]
])



alpha_by_agegroup = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.3}
proportion_of_agegroup = {"0-29": 0.1, "30-59": 0.3, "60-89": 0.4, "89+": 0.2}
s = 0.01
alpha_opt = sum(alpha_by_agegroup[i] * proportion_of_agegroup[i] for i in list(alpha_by_agegroup.keys()))



for j in numpy.arange(entrada.shape[0]):

	cidade=entrada[j][0]
	nome_cidade=cidade
	nome_cidade=re.sub(r'-',r' ',str(nome_cidade))
	populacao=entrada[j][1].astype('int')

	arquivo="../../boletins/"+cidade+"/resumo-"+cidade+".csv"
	dados=numpy.genfromtxt(arquivo,delimiter=";",dtype=int)
	texto=numpy.genfromtxt(arquivo,delimiter=";",dtype=str)
	z=numpy.where(dados.T[1]>=0)[0]
	data=dados.T[1][z]
	confirm=dados.T[5][z]
	obitos=dados.T[6][z]
	#recup=dados.T[8][z]
	
	L=None
	#L=z.size+7  #Lockdown decretado 7 dias apos o presente dia
	#L=30  #Rio de Janeiro
	#L=50  #Campos dos Goytacazes
	
	
	N = populacao
	D = 7.0 # infections lasts four days
	gamma = 1.0 / D
	delta = 1.0 / 7.0  # incubation period of five days

	#R_0_start, R_0_end = entrada[j][4:].astype(float)
	#coeff, x0 = 0.2, 90
	R_0_start, coeff, x0, R_0_end = 2.8, 0.2, 90, 0.9
	#R_0_start, coeff, x0, R_0_end = 2.8, 0.3, 56.3, 1.1 # Rio de Janeiro

	#alpha=alpha_opt
	alpha = 0.02  # 2% constant death rate
	
	rho = 1.0/9.0  # 9 days from infection until death
	#S0, E0, I0, R0, D0 = N-1, 8, 4, 0, 0  # initial conditions: one exposed
	E0, I0=entrada[j][2:4].astype(int) # initial conditions: one exposed
	S0=N-E0
	R0=D0=0

	#t = numpy.linspace(0, 100, 100) # Grid of time points (in days)
	t = numpy.arange(201) # Grid of time points (in days)
	y0 = S0, E0, I0, R0, D0 # Initial conditions vector

	#R_0_over_time = numpy.array([logistic_R_0(i) for i in range(len(t))])  # to plot R_0 over time: get function values
	R_0_over_time = logistic_R_0(t)

	# Integrate the SIR equations over the time grid, t.
	
	if( alpha != alpha_opt ):
		ret = odeint(deriv1, y0, t, args=(N, beta, gamma, delta, alpha, rho))
		S, E, I, R, D = ret.T
		plotseird(t, S, E, I, R, D, L=L, R_0=R_0_over_time,Alpha=alpha)
	else:
		ret = odeint(deriv2, y0, t, args=(N, beta, gamma, delta, alpha_opt, rho))
		S, E, I, R, D = ret.T
		Alpha_over_time = [s * I[i]/N + alpha_opt for i in range(len(t))]  # to plot alpha over time
		plotseird(t, S, E, I, R, D, L=L, R_0=R_0_over_time,Alpha=Alpha_over_time)
		
	
	#subnotificacao()
	
	print nome_cidade," & ",int(I.max())," & ",int(I.max()*0.12)," & ",int(S[-1])," & ",int(R[-1])," & ",int(D[-1])," \\\\"
