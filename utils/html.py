import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br, video, source, attr
from dominate.tags import span
import os


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and
     manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be
            created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            reflect (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_videos(self, vids, txts, links, width=400, hidden_tag="hidden"):
        """add images to the HTML file

        Parameters:
            vids (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image,
            it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        colors = ["red", "blue", "gold", "salman"]
        with self.t:
            with tr():
                for vid, txt, link in zip(vids, txts, links):
                    td_style = "word-wrap: break-word; width:{}px".format(width)
                    with td(style=td_style, halign="center", valign="top"):
                        with p():
                            vid_path = str(vid)
                            if vid_path == hidden_tag:
                                p_style = "font-weight: bold; width:{}px;"
                                p_style = p_style.format(width * 3)
                                p("hidden video", style=p_style)
                            else:
                                with a(href=str(link)):
                                    with video():
                                        attr(controls="controls")
                                        source(src=vid_path, type="video/mp4")
                            br()
                            rows = txt.split("<br>")
                            for idx, row in enumerate(rows):
                                color = colors[idx % len(colors)]
                                bold_tag = "<b>"
                                if not row.startswith(bold_tag):
                                    s_style = "color:{};".format(color)
                                else:
                                    s_style = "color:black; font-weight: bold;"
                                    row = row[len(bold_tag):]
                                span(row, style=s_style)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image,
            it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        colors = ["red", "blue", "gold", "salman"]
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    td_style = "word-wrap: break-word;"
                    with td(style=td_style, halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(
                                    style="width:%dpx" % width,
                                    src=im,
                                )
                            br()
                            rows = txt.split("<br>")
                            for idx, row in enumerate(rows):
                                color = colors[idx % len(colors)]
                                bold_tag = "<b>"
                                if not row.startswith(bold_tag):
                                    s_style = "color:{};".format(color)
                                else:
                                    s_style = "color:black; font-weight: bold;"
                                    row = row[len(bold_tag):]
                                span(row, style=s_style)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = "%s/index.html" % self.web_dir
        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


if __name__ == "__main__":  # we show an example usage here.
    html = HTML("web/", "test_html")
    html.add_header("hello world")

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append("image_%d.png" % n)
        txts.append("text_%d" % n)
        links.append("image_%d.png" % n)
    html.add_images(ims, txts, links)
    html.save()