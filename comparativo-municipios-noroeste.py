#!/usr/bin/python


import pylab
import numpy
from datetime import date
import scipy.optimize
import re

############################################
def func1(x, a, b):
	return a*numpy.exp(b*x)
	
############################################
def func2(x, a, b):
	return a*x+b
	
############################################

def graph1(xlim, ylim,c,u):
	p0=(1,0.1)
	#if(cidade[i]=="Pirapetinga"):
	#	p0=(-10**3,-10**-5,10**3)
	m=x
	n=y
	if(cidade[i]=="Campos-dos-Goytacazes"):
		m=x[:34]
		n=y[:34]
		popt, pcov = scipy.optimize.curve_fit(func1, m,n,  p0=p0)
		xf=numpy.linspace(x[0],xlim*3,200)
		yf=func1(xf,popt[0],popt[1])
		if((j==2)and (u==1)):
			pylab.plot(xf-init,yf,'--',color='gray')
		m=numpy.append(x[:10],x[34:])
		n=numpy.append(y[:10],y[34:])

	popt, pcov = scipy.optimize.curve_fit(func1, m, n,  p0=p0)
	duplic=0.0
	if(j==1):
		print nome_cidade[i]+" & {:.6} & {:.5} \\\\ % exponencial".format(popt[0],popt[1])
		duplic=numpy.log(2)/popt[1]

		#print pcov
	elif((j==2) and (u==1)):
		next_weeks=numpy.array([7,14,21,28])+x[-2]
		duplic=popt[0]*numpy.exp(popt[1]*next_weeks)
		#pylab.text(41,53,"Coeficientes do ajuste exponencial A*exp(B*x)")
		#pylab.text(41,pos,str(nome_cidade[i])+": ")
		##"A= %.3f , B= %.3f" % str(popt[0]),str(popt[1])
		#pylab.text(50,pos,'A= {:2.5f} , '.format(popt[0]))
		#pylab.text(55,pos,'B= {:2.5f}'.format(popt[0]))

		#pylab.text(11.5,76.5,"Coeficientes do ajuste exponencial A*exp(B*x)")
		#pylab.text(11.5,pos,'A= {:2.5f} , '.format(popt[0]))
		#pylab.text(16.5,pos,'B= {:2.5f}'.format(popt[1]))
	elif(j==3):
		m=y.astype(float)/populacao[i]
		print nome_cidade[i],' & {:9.6f} \\% \\\\ % incidencia'.format(m[-2]*100)
		popt, pcov = scipy.optimize.curve_fit(func1, x, m,  p0=p0)
		
	xf=numpy.linspace(x[0],xlim*3,200)
	yf=func1(xf,popt[0],popt[1])
	pylab.plot(xf-init,yf,'--',color=c)
	return duplic

############################################

def plot_boletins():
	pylab.figure(4,figsize=(16,9))
	pylab.clf()
	temp=re.sub('-',' ',cidade[i])
	pylab.title("Dados dos boletins epidemiologicos COVID-19 de "+temp)
	
	pylab.plot(x-init,dados.T[2],'o',color='y',label='Casos Notificados')
	pylab.plot(x-init,dados.T[3],'o',color='g',label='Casos Suspeitos')
	pylab.plot(x-init,dados.T[4],'o',color='m',label='Casos com coleta')
	pylab.plot(x-init,dados.T[6],'o',color='c',label='Obitos Confirmados')
	pylab.plot(x-init,dados.T[7],'o',color='r',label='Casos Descartados',markersize=9)
	pylab.plot(x-init,dados.T[8],'o',color='k',label='Casos Curados')
	pylab.plot(x-init,dados.T[10],'o',color='blueviolet',label='Total de Exames')
	pylab.plot(x-init,dados.T[11],'o',color='lightblue',label='Internacoes Hospitalares')
	pylab.plot(x-init,dados.T[12],'o',color='lightgreen',label='Isolamento domiciliar')
	pylab.plot(x-init,dados.T[13],'o',color='sienna',label='Monitorados')
	pylab.plot(x-init,dados.T[14],'o',color='darksalmon',label='Casos Graves')
	pylab.plot(x-init,dados.T[15],'o',color='turquoise',label='Obitos Investigados')
	pylab.plot(x-init,dados.T[16],'o',color='pink',label='Sindrome Gripal')
	pylab.plot(x-init,dados.T[17],'o',color='indigo',label='SRAG')
	
	pylab.plot(x-init,y,'o',color='b',label='Casos Confirmados')
	delta=10
	#graph1(xx[-1]+delta,yy[-1]+delta,'b',0)

	
	pylab.xlim(0,xx[-1]+2)
	pylab.ylim(0,dados.T[2:10].max()+2)
	#pylab.xlim(x[0],xx[-1]+delta)
	#pylab.ylim(0,yy[-1]+delta)
	pylab.xticks(x,d,rotation='vertical')
	#pylab.yticks(numpy.arange(0,y.max()+1,int(float(y.max()+delta)/10)))
	pylab.xlabel("Datas de emissao de boletins epidemiologicos")
	pylab.ylabel("Informacoes registradas nos boletins")
	pylab.legend(loc='upper left')
	pylab.subplots_adjust(bottom=0.20)
	pylab.grid()
	pylab.show(block=False)
	pylab.savefig('boletins-municipio-'+str(cidade[i])+'.pdf',dpi=200)
	pylab.figure(j,figsize=(16,9))

############################################

def plot_ajustes_exponenciais():
	pylab.figure(5,figsize=(16,9))
	pylab.clf()
	temp=re.sub(r'-',r' ',cidade[i])
	pylab.title("Ajuste exponencial de casos confirmados para "+temp)
	delta=1
	pylab.annotate('$1^o$ caso',xy=(0,3), xycoords='data',xytext=(xx[0], float(yy.max()+8)/3), textcoords='data',arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='bottom', verticalalignment='top')
	
	#m=re.sub(r',',r'.',media)
	#b=float(xx[-1])/float(m)
	#a=numpy.arange(xx[0],int(b)+4)
	#x_double=a*float(m)
	#a=numpy.zeros(x_double.size)+2
	#b=numpy.arange(x_double.size)
	#y_double=(a**b)*yy[0]
	#popt, pcov = scipy.optimize.curve_fit(func1, x_double, y_double, p0=(1,0.1,0))
	#xf=numpy.linspace(x[0],x_double[-1],200)
	#yf=func1(xf,popt[0],popt[1],popt[2])
	#pylab.plot(xf,yf,'--',color='gray')
	#pylab.plot(x_double,y_double,'x',color='k')
	
	pylab.plot(x-init,y,'o',color='b',label='Casos Confirmados')
	h=graph1(xx[-1]+delta,yy[-1]+delta,'b',1)
	pylab.xlim(x[0]-delta,xx[-1]+delta)
	pylab.ylim(0,yy[-1]+delta)
	pylab.xticks(x,d,rotation='vertical')
	pylab.xlabel("Datas de emissao de boletins epidemiologicos")
	pylab.ylabel("Numero de casos confirmados")
	pylab.legend(loc='upper left')
	pylab.subplots_adjust(bottom=0.20)
	pylab.grid()
	pylab.show(block=False)
	pylab.savefig('ajustes-exponenciais-'+str(cidade[i])+'.pdf',dpi=200)
	pylab.figure(j,figsize=(16,9))
	return h

############################################

def plot_tempo_duplicacao():
	pylab.figure(6,figsize=(16,9))
	pylab.clf()
	temp=re.sub('-',' ',cidade[i])
	pylab.title("Tempo de duplicacao de casos confirmados em "+temp)
	
	temp=dados.T[18]
	w=numpy.where(temp>0)[0]
	#print "Tempo de duplicacao\t",cidade[i],temp[w].mean()
	
	pylab.plot(x[w],temp[w],'o-',color='r',label='Tempo de duplicacao')
	pylab.plot(x[w],numpy.zeros(w.size)+temp[w].mean(),'--',color='k',linewidth=2,label='Tempo de duplicacao')

	coefs=numpy.polyfit(x[w],temp[w],1)
	p=numpy.poly1d(coefs)
	print nome_cidade[i]+" & {:.6} & {:.6} \\\\ % reta".format(coefs[0],coefs[1])
	xf=numpy.linspace(x[w][0],x[w][-1],20)
	yf=p(xf)
	pylab.plot(xf,yf,'-',linewidth=2,color='gray')

	delta=10
	#graph1(xx[-1]+delta,yy[-1]+delta,'b',0)	
	pylab.xlim(0,xx[-1]+2)
	pylab.ylim(0,temp.max()+2)
	#pylab.xlim(x[0],xx[-1]+delta)
	#pylab.ylim(0,yy[-1]+delta)
	pylab.xticks(x[w],d[w],rotation='vertical')
	#pylab.yticks(numpy.arange(0,y.max()+1,int(float(y.max()+delta)/10)))
	pylab.xlabel("Datas de medicao da duplicacao de casos")
	pylab.ylabel("Tempo de duplicacao")
	pylab.legend(loc='upper left')
	pylab.subplots_adjust(bottom=0.20)
	pylab.grid()
	pylab.show(block=False)
	pylab.savefig('tempo-duplicacao-'+str(cidade[i])+'.pdf',dpi=200)
	pylab.figure(j,figsize=(16,9))

############################################

#Nome da cidade, populacao estimada, cor adotada
entrada=numpy.array([
#["Rio-de-Janeiro",6718903,"cyan"],
["Campos-dos-Goytacazes",507548,"y"],
["Itaocara",23234,"red"],
["Itaperuna",103224,"green"],
["Miracema",27174,"orange"],
["Pirapetinga",10752,"magenta"],
["Santo-Antonio-de-Padua",42479,"blue"],
["Sao-Fidelis",38669,"black"],
])

#entrada=numpy.array([["Rio-de-Janeiro",6718903,"blue"]])

cidade=entrada.T[0]
nome_cidade=[re.sub(r'-',r' ',cid) for cid in cidade]
populacao=entrada.T[1].astype('int')
color=entrada.T[2]
output=numpy.empty((0,4),dtype=str)
output2=numpy.empty((0,1),dtype=str)
hoje=date.today()


for j in [0,1,2,3]:
#for j in [3]:

	pylab.figure(j,figsize=(16,9))
	pylab.clf()
	pylab.title("Evolucao de casos de COVID-19 nos municipios do Noroeste Fluminense em "+str(hoje))
	if(j<=1):
		pos=32.3
	else:
		#pos=50
		pos=73.55

	for i in numpy.arange(cidade.size):

		arquivo="../boletins/"+cidade[i]+"/resumo-"+cidade[i]+".csv"
		dados=numpy.genfromtxt(arquivo,delimiter=";",dtype=int)
		texto=numpy.genfromtxt(arquivo,delimiter=";",dtype=str)
		index=numpy.where(dados.T[5]>0)
		init=dados.T[1][index][0]
		data=texto[index][0][0]
		media=texto[-1][-1]
		
		#temp,x,notif,susp,coleta,y,ob,desc,cur,ex,intern,isol,monit,graves,obitinvest,sgripal,srag=dados.T[:17]
		d=texto.T[0]
		x=dados.T[1]
		y=dados.T[5]
		#print cidade[i],i,j
		#print x,"<<<<<<<<<<<<<<"
		xx=dados.T[1][index]-init
		yy=dados.T[5][index]
		if(j==3):
			yy=yy.astype(float)/populacao[i]
		
		pylab.plot(xx,yy,'o-',color=color[i],label=cidade[i])
		
		xlim=55
		ylim=420
		if(j==0):
			#pylab.text(7,33.5,"Primeiro caso  |  Dobra casos a cada")
			#pylab.text(7, pos, str(data)+"           "+str(media)+" dias")
			pos=pos-1.15
		elif (j==1):
			#pylab.text(7,33.5,"Primeiro caso  |  Dobra casos a cada")
			#pylab.text(7, pos, str(data)+"           "+str(media)+" dias")
			#pos=pos-1.15
			xlim=45
			ylim=160
			duplic=graph1(xlim,ylim,color[i],0)
			o=numpy.array([[nome_cidade[i],str(data),str(media),str(duplic)]],dtype=str)
			output=numpy.concatenate((output,o),axis=0)

		elif(j==2):
			xlim=60
			ylim=300
			graph1(xlim,ylim,color[i],0)
			pos-=2.55
			plot_boletins()
			h=plot_ajustes_exponenciais()
			o=nome_cidade[i]+" & {:10.5} & {:10.5}  \\\\ % projecao de casos".format(h[0],h[1])
			output2=numpy.append(output2,o)
			if(cidade[i]!='Rio-de-Janeiro'):
				plot_tempo_duplicacao()
		elif(j==3):
			xlim=60
			ylim=4*10**-3
			graph1(xlim,ylim,color[i],1)

	pylab.xlim(0,xlim)
	pylab.ylim(0,ylim)
	pylab.xticks(numpy.arange(0,xlim+1,5))
	pylab.yticks(numpy.arange(0,ylim+1,30))
	if(j==3):
		vals=numpy.linspace(0,ylim,13)
		pylab.yticks(vals,['{:,.2%}'.format(x) for x in vals])
		pylab.title('Incidencia de COVID-19 na populacao em '+str(hoje))
	
	pylab.grid()
	pylab.xlabel("Numero de dias apos o primeiro caso")
	pylab.ylabel("Numero de casos confirmados")
	pylab.legend(bbox_to_anchor=(0, 0.96),loc='upper left',ncol=1)
	pylab.show(block=False)
	pylab.savefig('figura-cidades-noroeste-'+str(j)+'.pdf',dpi=200)

print "\n\n",output2
print "\n\n",output

